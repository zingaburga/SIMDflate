#include "../include/simdflate.h"
#include <vector>

#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
	#define IS_X64 1
#endif
#if defined(IS_X64) && \
	(defined(_MSC_VER) && _MSC_VER >= 1920 && !defined(__clang__)) || \
	(defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512VBMI__) && defined(__AVX512VBMI2__) && defined(__AVX512BW__) && defined(__AVX512VNNI__) && defined(__AVX512VPOPCNTDQ__) && defined(__VPCLMULQDQ__) && defined(__AVX512IFMA__) && defined(__GFNI__) && defined(__PCLMUL__) && defined(__LZCNT__) && defined(__BMI__) && defined(__BMI2__))

const int SIMDFLATE_COMPILER_SUPPORTED = 1;

#include "common.hh"
#include "checksum.hh"
#include "bitwriter.hh"
#include "huffmantree.hh"
#include "lz77data.hh"
#include "lz77.hh"
#include "histcount.hh"
#include "huffmanenc.hh"


// write the dynamic Huffman block header [https://www.w3.org/Graphics/PNG/RFC-1951#dyn]
static void write_dyn_block_header(BitWriter& output, const HuffmanTree<286, 15>& huf_litlen, const HuffmanTree<30, 15>& huf_dist, bool is_last_block) {
	uint8_t table[NUM_SYM_TOTAL];   // pre-Huffman encoded Code Length codes
	uint16_t codelen_hist[19] = {}; // histogram of above table
	unsigned num_litlen = 257, num_dist = 1; // number of litlen/distance codes, initially set to minimum allowed values
	
	// generate Code Length data and collect histogram
	// note that num_litlen & num_dist will get updated
	auto table_len = huf_litlen.WriteLengthTable(table, codelen_hist, num_litlen);
	table_len += huf_dist.WriteLengthTable(table + table_len, codelen_hist, num_dist);
	
	// generate Huffman tree for the Code Length data
	alignas(64) HuffmanTree<19, 7> huf_codelen(codelen_hist);
	
	
	auto huf_codelen_lut = _mm256_load_si256(reinterpret_cast<const __m256i*>(huf_codelen.lengths));
	// re-arrange lengths to DEFLATE specified order
	auto codelen_lengths = _mm256_permutexvar_epi8(_mm256_set_epi8(
		31,31,31,31,31,31,31,31,31,31,31,31,31, // these are guaranteed to be 0s
		15, 1, 14, 2, 13, 3, 12, 4, 11, 5, 10, 6, 9, 7, 8, 0, 18, 17, 16
	), huf_codelen_lut);
	
	// max 7 bits per Code Length symbol
	assert(_mm256_cmpge_epu8_mask(codelen_lengths, _mm256_set1_epi8(8)) == 0);
	
	// strip zeroes from codelen
	auto nonzero_len = _cvtmask32_u32(_mm256_test_epi8_mask(codelen_lengths, codelen_lengths));
	nonzero_len |= 0b1000; // minimum length is 4
	int num_codelen = 32 - _lzcnt_u32(nonzero_len);
	
	assert(num_litlen >= 257);
	assert(num_dist >= 1);
	assert(num_codelen >= 4);
	
	// write the first part of the header
	output.ZeroWrite57(
		(is_last_block ? 1:0)
		| (2 << 1) // dynamic Huffman [2]
		| ((num_litlen - 257) << 3)  // [5]
		| ((num_dist - 1) << 8) // [5]
		| ((num_codelen - 4) << 13)  // [4]
		, 1+2+5+5+4
	);
	// write the lengths for Code Length alphabet
	auto packed_codelen_lengths = _mm256_cvtepi16_epi8(_mm256_maddubs_epi16(codelen_lengths, _mm256_set1_epi16(0x1001)));
	const uint64_t THREE_BITS = 0x7777777777777777ULL;
	uint64_t codelen_length_data =
		(uint64_t(_pext_u32(_mm_extract_epi32(packed_codelen_lengths, 2), uint32_t(THREE_BITS))) << 48)
		| _pext_u64(_mm_cvtsi128_si64(packed_codelen_lengths), THREE_BITS);
	output.ZeroWrite57(codelen_length_data, num_codelen*3); // up to 57 bits
	
	// compute Huffman code for Code Length
	alignas(64) uint16_t codelen_codes[19 +13];
	huf_codelen.CalcCodes(codelen_codes);
	
	// Huffman encode Code Lengths and write it out
	auto huff_sym_lut = _mm512_load_si512(codelen_codes);
	assert(_mm512_test_epi16_mask(huff_sym_lut, _mm512_set1_epi16(0xff00)) == 0);
	huff_sym_lut = _mm512_castsi256_si512(_mm512_cvtepi16_epi8(huff_sym_lut));
	auto last_data = _mm512_setzero_si512();
	loop_u8x64(table_len, table, [&](__m512i data, uint64_t is_valid, size_t) {
		auto is_valid_mask = _cvtu64_mask64(is_valid);
		
		// if the top bit of `data` is set, it indicates a repeat length (i.e. not a symbol)
		//auto is_sym = _kandn_mask64(_mm512_movepi8_mask(data), is_valid_mask);
		auto is_sym = _mm512_mask_testn_epi8_mask(is_valid_mask, data, _mm512_set1_epi8(-128));
		// get Huffman codes, and remove top bit from repeat lengths
		auto huff_data = _mm512_mask_permutexvar_epi8(
			_mm512_and_si512(data, _mm512_set1_epi8(0x7f)), // repeat lengths
			is_sym, data, huff_sym_lut  // symbols
		);
		// compute bit lengths for repeat codes
		auto data_shift = _mm512_alignr_epi8(data, _mm512_alignr_epi64(data, last_data, 6), 15);
		auto repeat_bitlen = _mm512_maskz_shuffle_epi8(is_valid_mask, _mm512_set4_epi32(0,0,0, 0x00070302), data_shift);
		// get bit lengths of Huffman codes + merge with above
		auto huff_len = _mm512_mask_permutexvar_epi8(
			repeat_bitlen, is_sym, data, _mm512_castsi256_si512(huf_codelen_lut)
		);
		last_data = data;
		
		// we now have all the Huffman encoded symbols, just need to join them together
		auto emptylen = _mm512_sub_epi8(_mm512_set1_epi8(8), huff_len); // TODO: see if this can be eliminated
		// combine 2x 8b into 16b
		// note that double-shifts are used instead of single-shifts, due to how they differ in interpreting the shift amount (double-shift ignores irrelevant bits, whilst single-shift considers the whole element)
		// TODO: see which of these two is better
		/*
		auto lomul = _mm512_shuffle_epi8(
			_mm512_set4_epi32(0, 1, 0x02040810, 0x20408000), huff_len
		);
		huff_data = _mm512_mask_gf2p8mul_epi8(huff_data, MASK_ALTERNATE, huff_data, lomul);
		*/
		huff_data = _mm512_mask_blend_epi8(
			MASK_ALTERNATE, huff_data,
			_mm512_shldv_epi16(huff_data, _mm512_setzero_si512(), emptylen)
		);
		
		huff_data = _mm512_shrdv_epi16(huff_data, _mm512_setzero_si512(), emptylen);
		emptylen = _mm512_maddubs_epi16(emptylen, _mm512_set1_epi8(1));
		
		// 4x 16b -> 64b
		huff_data = huffman_bitjoin_16_64(huff_data, emptylen);
		emptylen = _mm512_sad_epu8(emptylen, _mm512_setzero_si512());
		
		// 8x 64b -> 512b
		int total_len;
		huff_data = huffman_bitjoin_64_512(huff_data, emptylen, total_len);
		
		output.ZeroWrite505(huff_data, total_len);
	});
}

template<class ChecksumClass>
static size_t deflate_main(void* HEDLEY_RESTRICT dest, const void* HEDLEY_RESTRICT src, size_t len, ChecksumClass& cksum) {
	// don't bother compressing short messages
	if(len < 64) { // TODO: adjust threshold for no compression, don't allow it to be larger than 256 though
		if(len == 0) {
			// write blank header
			const uint16_t BLANK_HEADER = 
				1  // last block [1]
				| (1 << 1) // static Huffman [2]
				| (0 << 3) // symbol 256 (end of block) [7]
			;
			memcpy(dest, &BLANK_HEADER, 2);
			return 2;
		}
		assert(len < 65536); // size limit of uncompressed block
		uint64_t header = 
			1  // last block [1]
			| (0 << 1)   // no compression [2]
			// 5 bits of padding, to byte align
			| (len << 8) // length
			| ((~len) << 24) // NLEN
		; // 40 bits total
		memcpy(dest, &header, 5);
		
		auto dest_ = static_cast<uint8_t*>(dest) + 5;
		auto src_ = static_cast<const uint8_t*>(src);
		
		size_t pos = 0;
		for(; pos + sizeof(__m512i) <= len; pos += sizeof(__m512i)) {
			auto data = _mm512_loadu_si512(src_ + pos);
			cksum.update(data);
			_mm512_storeu_si512(dest_ + pos, data);
		}
		
		if(pos < len) { // final vector
			auto loadmask = _bzhi_u64(-1LL, len-pos);
			auto data = _mm512_maskz_loadu_epi8(_cvtu64_mask64(loadmask), src_ + pos);
			cksum.update_partial(data, len-pos);
			
			auto storemask = (loadmask + 1) | loadmask; // set an extra mask bit for the trailing 3 bits
			_mm512_mask_storeu_epi8(dest_ + pos, _cvtu64_mask64(storemask), data);
		}
		
		return 5 + len;
	}
	
	BitWriter output(dest);
	
	std::vector<uint32_t> match_offsets(1 << MATCH_TABLE_ORDER); // LZ77 match indices
	bool first_block = true;
	uint16_t window_offset = 0;
	unsigned skip_next_bytes = 0;
	while(len) {
		// LZ77 encode data
		alignas(64) Lz77Data lz77data;
		lz77_encode(lz77data, src, len, window_offset, skip_next_bytes, match_offsets.data(), first_block, cksum);
		
		// histogram LZ77 output
		// TODO: consider idea of doing symbol count before lz77, and using the information to assist lz77 in using a match or not?
		alignas(64) uint16_t sym_counts[NUM_SYM_TOTAL];
		bool is_sample = symbol_histogram(sym_counts, lz77data);
		// TODO: consider dynamic block splitting in symbol_histogram?
		
		// generate Huffman trees for litlen/distance alphabets
		alignas(64) HuffmanTree<286, 15> huf_litlen(sym_counts, is_sample);
		alignas(64) HuffmanTree<30, 15> huf_dist(sym_counts + 288, is_sample);
		
		// write block header
		write_dyn_block_header(output, huf_litlen, huf_dist, len==0);
		// TODO: compute size of uncompressed block, and if above block is larger, rewrite it as an uncompressed block?
		// - if sampling symbol counts, can only do the check after Huffman encoding
		
		// encode data to output
		huffman_encode(output, huf_litlen, huf_dist, lz77data);
		
		// proceed to next block
		first_block = false;
	}
	
	return output.Length(dest);
}


// encode as raw DEFLATE
size_t simdflate_deflate(void* dest, const void* src, size_t len) {
	Checksum_None cksum;
	auto outlen = deflate_main(dest, src, len, cksum);
	_mm256_zeroupper();
	return outlen;
}

// encode with zlib wrapper
size_t simdflate_zlib(void* dest, const void* src, size_t len) {
	auto dest_u8 = static_cast<uint8_t*>(dest);
	
	uint16_t header = 
		8 // CM = deflate
		| ((WINDOW_ORDER-8) << 4) // CINFO
		| (0 << 13) // no preset dictionary
		| (0 << 14) // FLEVEL = 'fastest' compression level used
	;
	// add in FCHECK
	header |= (31 - (((header >> 8) | (header << 8)) % 31)) << 8;
	memcpy(dest_u8, &header, sizeof(header));
	dest_u8 += 2;
	
	// TODO: checksum vectors don't stay in registers; maybe the compiler doesn't know how long it needs to hold onto them, so it assumes it's needed throughout the entire process, so spills them.  Investigate finalising the checksum after LZ77 completes, so compiler doesn't assume it needs to be remembered throughout the process
	
	Checksum_Adler32 cksum;
	auto out_len = deflate_main(dest_u8, src, len, cksum);
	cksum.finalise(dest_u8 + out_len);
	
	_mm256_zeroupper();
	return out_len + 6;
}

// encode with basic gzip wrapper
size_t simdflate_gzip(void* dest, const void* src, size_t len) {
	auto dest_u8 = static_cast<uint8_t*>(dest);
	
	const uint16_t magic = 0x8b1f;
	memcpy(dest_u8, &magic, sizeof(magic));
	const uint64_t header = 
		8 // deflate compression
		| (0 << 8) // header flags
		| (0 << 16) // no MTIME
		| (4ULL << 48) // compression flag (fastest algorithm used)
		| (255ULL << 56) // unknown OS
	;
	memcpy(dest_u8+2, &header, sizeof(header));
	dest_u8 += 10;
	
	Checksum_Crc32 cksum;
	auto out_len = deflate_main(dest_u8, src, len, cksum);
	cksum.finalise(dest_u8 + out_len);
	// write length
	memcpy(dest_u8 + out_len + 4, &len, sizeof(uint32_t));
	
	_mm256_zeroupper();
	return out_len + 18;
}

#else

const int SIMDFLATE_COMPILER_SUPPORTED = 0;

#ifdef IS_X64
HEDLEY_WARNING("Compiler doesn't have necessary AVX-512 support for SIMDflate, or appropriate options haven't been enabled");
#endif

// dummy functions so that it compiles
size_t simdflate_deflate(void*, const void*, size_t) { return 0; }
size_t simdflate_zlib(void*, const void*, size_t) { return 0; }
size_t simdflate_gzip(void*, const void*, size_t) { return 0; }

#endif
