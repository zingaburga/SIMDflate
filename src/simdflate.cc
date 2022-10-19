#ifdef SIMDFLATE_DBG_DUMP_LZ77
#define _CRT_SECURE_NO_WARNINGS 1
#include <cstdio>
#include <cerrno>
#endif

#include "../include/simdflate.h"
#include <vector>

#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
	#define IS_X64 1
#endif
#if defined(IS_X64) && \
	(defined(_MSC_VER) && _MSC_VER >= 1920 && !defined(__clang__)) || \
	(defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512VBMI__) && defined(__AVX512VBMI2__) && defined(__AVX512BW__) && defined(__AVX512BITALG__) && defined(__AVX512VNNI__) && defined(__AVX512VPOPCNTDQ__) && defined(__VPCLMULQDQ__) && defined(__AVX512IFMA__) && defined(__AVX512CD__) && defined(__GFNI__) && defined(__PCLMUL__) && defined(__LZCNT__) && defined(__BMI__) && defined(__BMI2__))

const int SIMDFLATE_COMPILER_SUPPORTED = 1;

#include "common.hh"
#include "checksum.hh"
#include "bitwriter.hh"
#include "huffmantree.hh"
#include "lz77data.hh"
#include "lz77.hh"
#include "histcount.hh"
#include "huffmanenc.hh"

// length of a raw (uncompressed) DEFLATE block(s)
static inline unsigned raw_block_len(size_t src_len, int output_bitpos) {
	assert(src_len > 0);
	assert(output_bitpos >= 0 && output_bitpos < 8);
	
	unsigned raw_len = (8 - (output_bitpos + 3)) & 7; // bits reserved for padding
	size_t cur_block_len = std::min(src_len, size_t(65535));
	raw_len += 32 + cur_block_len*8;
	src_len -= cur_block_len;
	// add size of remaining blocks
	raw_len += 40*(src_len/65535) + src_len*8;
	return raw_len;
}

// compute length of a dynamic + fixed Huffman block, based on symbol histogram
static void calc_huffman_len(const HuffmanTree<286, 15>& huf_litlen, const HuffmanTree<30, 15>& huf_dist, const uint16_t* sym_counts, unsigned& dyn_size, unsigned& fixed_size) {
	assert(sym_counts[286] == 0 && sym_counts[287] == 0 && sym_counts[318] == 0 && sym_counts[319] == 0);
	
	__m512i dyn_lo, dyn_hi, fixed_lo, fixed_hi;
	dyn_lo = dyn_hi = fixed_lo = fixed_hi = _mm512_setzero_si512();
	auto accumulate = [&](int i, const uint8_t* huf_len, __m512i fixed_len, __m512i xbits_len) {
		auto hist = _mm512_load_si512(sym_counts + i);
		auto dyn_len = _mm512_cvtepu8_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(huf_len)));
		
		// add in extra bits
		fixed_len = _mm512_add_epi16(fixed_len, xbits_len);
		dyn_len = _mm512_add_epi16(dyn_len, xbits_len);
		
		// split the sign bit off to workaround the lack of an unsigned multiply
		auto hist_lo = _mm512_and_si512(hist, _mm512_set1_epi16(0x7fff));
		auto hist_hi = _mm512_srli_epi16(hist, 15);
		
		dyn_lo = _mm512_dpwssd_epi32(dyn_lo, hist_lo, dyn_len);
		dyn_hi = _mm512_dpwssd_epi32(dyn_hi, hist_hi, dyn_len);
		fixed_lo = _mm512_dpwssd_epi32(fixed_lo, hist_lo, fixed_len);
		fixed_hi = _mm512_dpwssd_epi32(fixed_hi, hist_hi, fixed_len);
	};
	accumulate(  0, huf_litlen.lengths +   0, _mm512_set1_epi16(8), _mm512_setzero_si512());
	accumulate( 32, huf_litlen.lengths +  32, _mm512_set1_epi16(8), _mm512_setzero_si512());
	accumulate( 64, huf_litlen.lengths +  64, _mm512_set1_epi16(8), _mm512_setzero_si512());
	accumulate( 96, huf_litlen.lengths +  96, _mm512_set1_epi16(8), _mm512_setzero_si512());
	accumulate(128, huf_litlen.lengths + 128, VEC512_16(8 + _x/16), _mm512_setzero_si512());
	accumulate(160, huf_litlen.lengths + 160, _mm512_set1_epi16(9), _mm512_setzero_si512());
	accumulate(192, huf_litlen.lengths + 192, _mm512_set1_epi16(9), _mm512_setzero_si512());
	accumulate(224, huf_litlen.lengths + 224, _mm512_set1_epi16(9), _mm512_setzero_si512());
	const auto XBITS_LEN = huffman_lendist_xbits_len();
	accumulate(256, huf_litlen.lengths + 256, VEC512_16(_x >= 24 ? 8 : 7), _mm512_cvtepu8_epi16(_mm512_castsi512_si256(XBITS_LEN)));
	accumulate(288, huf_dist.lengths, _mm512_set1_epi16(5), _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(XBITS_LEN, 1)));

	// merge lo/hi halves
	dyn_lo = _mm512_add_epi32(dyn_lo, _mm512_slli_epi32(dyn_hi, 15));
	fixed_lo = _mm512_add_epi32(fixed_lo, _mm512_slli_epi32(fixed_hi, 15));
	
	// horizontal sum down
	auto huff_size = _mm512_add_epi32(
		_mm512_unpacklo_epi32(dyn_lo, fixed_lo),
		_mm512_unpackhi_epi32(dyn_lo, fixed_lo)
	);
	auto huff256 = _mm256_add_epi32(_mm512_castsi512_si256(huff_size), _mm512_extracti64x4_epi64(huff_size, 1));
	auto huff128 = _mm_add_epi32(_mm256_castsi256_si128(huff256), _mm256_extracti128_si256(huff256, 1));
	huff128 = _mm_add_epi32(huff128, _mm_unpackhi_epi64(huff128, huff128));
	
	dyn_size = _mm_cvtsi128_si32(huff128);
	fixed_size = _mm_extract_epi32(huff128, 1);
	
	// TODO: without the header, dynamic Huffman should almost never be larger than fixed Huffman (if package-merge is used, this should definitely be the case)
	assert(dyn_size > 0);
	assert(fixed_size > 0);
}

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
	int num_codelen = 32 - _lzcnt_u32(nonzero_len);
	assert(num_codelen >= 5); // minimum possible length requires at least one litlen/distance length
	
	assert(num_litlen >= 257);
	assert(num_dist >= 1);
	
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

// write headers for fixed Huffman or raw (uncompressed) blocks
static inline void write_fixed_block_header(BitWriter& output, bool is_last_block) {
	output.ZeroWrite57(2 + (is_last_block ? 1:0), 3);
}
static inline void write_raw_block_header(BitWriter& output, uint16_t len, bool is_last_block) {
	output.ZeroWrite57(is_last_block ? 1:0, 3);
	output.PadToByte();
	output.ZeroWrite57(uint32_t(len | (~len << 16)), 32);
}

template<class ChecksumClass>
static size_t deflate_main(void* HEDLEY_RESTRICT dest, const void* HEDLEY_RESTRICT src, size_t len, ChecksumClass& cksum) {
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
	
	BitWriter output(dest);
	
	std::vector<uint32_t> match_offsets(1 << MATCH_TABLE_ORDER); // LZ77 match indices
	bool first_block = true;
	uint16_t window_offset = 0;
	int skip_next_bytes = 0;
	auto remaining_len = len;
	while(remaining_len) {
		auto start_src = static_cast<const uint8_t*>(src);
		if(HEDLEY_UNLIKELY(remaining_len < 64)) { // no point in trying to compress really small messages
			if(remaining_len < 32) { // TODO: adjust threshold?
				// for length < 25, fixed Huffman is guaranteed to be more efficient over raw - this is because the raw header has a minimum 32 bit header, whilst fixed Huffman has a minimum 7 bit end-of-block symbol.
				write_fixed_block_header(output, true);
				
				auto mask = _cvtu64_mask64(_bzhi_u64(-1LL, remaining_len));
				auto data = _mm512_maskz_loadu_epi8(mask, src);
				huffman_fixed_encode_literals(output, data, mask);
				cksum.update_partial(data, remaining_len);
				output.ZeroWrite57(0, 7); // end-of-block
			} else {
				write_raw_block_header(output, remaining_len, true);
				auto dest_ = static_cast<uint8_t*>(output.Ptr());
				loop_u8x64(remaining_len, start_src, _mm512_setzero_si512(), [&](__m512i data, size_t pos) {
					cksum.update(data);
					_mm512_storeu_si512(dest_ + pos, data);
				}, [&](__m512i data, __mmask64 mask, size_t pos, size_t data_len) {
					cksum.update_partial(data, data_len);
					_mm512_mask_storeu_epi8(dest_ + pos, mask, data);
				});
				output.SkipBytes(remaining_len);
			}
			break;
		}
		
		// LZ77 encode data
		alignas(64) Lz77Data lz77data;
		lz77_encode(lz77data, src, remaining_len, window_offset, skip_next_bytes, match_offsets.data(), first_block, cksum);
		
#ifdef SIMDFLATE_DBG_DUMP_LZ77
		if(first_block) {
			auto f = fopen(SIMDFLATE_DBG_DUMP_LZ77, "wb");
			if(!f) {
				fprintf(stderr, "Failed to open output file: %s\n", strerror(errno));
				exit(1);
			}
			if(fwrite(lz77data.data, 1, lz77data.len, f) != lz77data.len) {
				fprintf(stderr, "Failed to write to output file: %s\n", strerror(errno));
				fclose(f);
				return 1;
			}
			fclose(f);
		}
#endif
		
		// histogram LZ77 output
		// TODO: consider idea of doing symbol count before lz77, and using the information to assist lz77 in using a match or not?
		alignas(64) uint16_t sym_counts[NUM_SYM_TOTAL];
		bool is_sample = symbol_histogram(sym_counts, lz77data);
		// TODO: consider dynamic block splitting in symbol_histogram?
		
		// generate Huffman trees for litlen/distance alphabets
		alignas(64) HuffmanTree<286, 15> huf_litlen(sym_counts, is_sample);
		alignas(64) HuffmanTree<30, 15> huf_dist(sym_counts + 288, is_sample);
		
		auto pre_header_pos = output.BitLength(dest);
		write_dyn_block_header(output, huf_litlen, huf_dist, remaining_len==0);
		unsigned dyn_header_len = output.BitLength(dest) - pre_header_pos;
		assert(dyn_header_len >= 3 /*block header*/ +5+5+4 /*lengths*/ + 5*3 /*min 5 length symbols*/ + 2*(1+7)+1 /*257x litlen codelengths as 2x sym18 + sym8*/ +1 /*1x distance codelength*/);
		// i.e. min length of 50 bits (realistically not possible though, since one symbol isn't going to be encoded at 8 bits, and there's an end-of-block symbol to consider)
		
		
		size_t src_block_size = static_cast<const uint8_t*>(src) - start_src;
		unsigned raw_block_size = raw_block_len(src_block_size, pre_header_pos & 7);
		bool write_raw_data = false;
		if(is_sample) {
			// write a dynamic Huffman block, then compare it with the raw block size
			// note, this does mean that the following write could exceed the write buffer, if sized according to the maximum!
			huffman_dyn_encode(output, huf_litlen, huf_dist, lz77data);
			
			unsigned cur_block_size = output.BitLength(dest) - pre_header_pos;
			
			if(HEDLEY_UNLIKELY(raw_block_size < cur_block_size)) {
				output.Rewind(cur_block_size);
				write_raw_data = true;
			}
		} else {
			unsigned dyn_block_size, fixed_block_size;
			calc_huffman_len(huf_litlen, huf_dist, sym_counts, dyn_block_size, fixed_block_size);
			
#ifndef NDEBUG
			{ // verify calculated sizes are correct
				std::vector<uint8_t> test_buf(src_block_size*2 + sizeof(__m512i));
				BitWriter test_out(test_buf.data());
				huffman_dyn_encode(test_out, huf_litlen, huf_dist, lz77data);
				unsigned actual_size = test_out.BitLength(test_buf.data());
				assert(actual_size == dyn_block_size);
				
				test_out.Rewind(dyn_block_size);
				assert(test_out.BitLength(test_buf.data()) == 0);
				huffman_fixed_encode(test_out, lz77data);
				actual_size = test_out.BitLength(test_buf.data());
				assert(actual_size == fixed_block_size);
			}
#endif
			
			dyn_block_size += dyn_header_len - 3; // add in header, except for 3-bit block header
			
			// TODO: consider prefetching the match table (for subsequent LZ77 round) at some point
			if(HEDLEY_LIKELY(dyn_block_size < fixed_block_size && dyn_block_size < raw_block_size)) {
				huffman_dyn_encode(output, huf_litlen, huf_dist, lz77data);
			} else if(fixed_block_size < raw_block_size) {
				output.Rewind(dyn_header_len);
				write_fixed_block_header(output, remaining_len==0);
				huffman_fixed_encode(output, lz77data);
			} else {
				output.Rewind(dyn_header_len);
				write_raw_data = true;
			}
		}
		
		if(HEDLEY_UNLIKELY(write_raw_data)) {
			while(src_block_size) {
				auto cur_block_size = std::min(src_block_size, size_t(65535));
				src_block_size -= cur_block_size;
				write_raw_block_header(output, cur_block_size, remaining_len==0 && src_block_size==0);
				memcpy(output.Ptr(), start_src, cur_block_size);
				output.SkipBytes(cur_block_size);
				start_src += cur_block_size;
			}
		}
		
		// proceed to next block
		first_block = false;
	}
	
	auto total_size = output.Length(dest);
	assert(total_size + sizeof(__m512i) <= simdflate_max_deflate_len(len));
	return total_size;
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
		| ((WINDOW_ORDER < 8 ? 0 : WINDOW_ORDER-8) << 4) // CINFO
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

size_t simdflate_max_deflate_len(size_t len) {
	// the output buffer size is guaranteed to hold at least that many input bytes
	size_t max_blocks = (len+OUTPUT_BUFFER_SIZE-1) / OUTPUT_BUFFER_SIZE;
	
	// maximum possible length occurs when all blocks are uncompressed
	// uncompressed blocks are at most 65535 bytes long, and have a 5 byte header
	// our block splitting strategy is a little inefficient, such that an extra block could be created per OUTPUT_BUFFER_SIZE sized block
	// also add a vector's worth of bytes for padding purposes
	return len + (len/65535)*5 + max_blocks*5 + sizeof(__m512i);
}

