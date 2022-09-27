#ifndef __SIMDFLATE_HUFFMANENC_H
#define __SIMDFLATE_HUFFMANENC_H

#include "common.hh"
#include "bitwriter.hh"
#include "huffmantree.hh"
#include "lz77data.hh"


// concatenate 4x 16-bit elements into a 64-bit element, given each 16-bit element has emptylen unoccupied bits (i.e. 16 minus emptylen bits)
// updates emptylen to the number of bits in 2x 32-bit elements (saves an add operation over returning it for 64-bit elements)
// returns the concatenated vector
static inline __m512i huffman_bitjoin_16_64(__m512i data, __m512i& emptylen) {
	assert(_mm512_cmpgt_epu16_mask(emptylen, _mm512_set1_epi16(16)) == 0);
	
	// 16->32
	data = _mm512_mask_sllv_epi16(data, __mmask32(MASK_ALTERNATE), data, emptylen);
	data = _mm512_shrdv_epi32(data, _mm512_setzero_si512(), emptylen); // TODO: explicit zeroing is probably unnecessary - investigate further in all cases of joining
	emptylen = _mm512_madd_epi16(emptylen, _mm512_set1_epi16(1));
	
	// 32->64
	data = _mm512_mask_sllv_epi32(data, __mmask16(MASK_ALTERNATE), data, emptylen);
	data = _mm512_shrdv_epi64(data, _mm512_setzero_si512(), emptylen);
	//emptylen = _mm512_sad_epu8(emptylen, _mm512_setzero_si512());
	
	return data;
}

// concatenate all 64-bit elements, given each 64-bit element has emptylen unoccupied bits
// the total number of bits is placed in total_bits
static inline __m512i huffman_bitjoin_64_512(__m512i data, __m512i emptylen, int& total_bits) {
	assert(_mm512_cmpgt_epu64_mask(emptylen, _mm512_set1_epi64(64)) == 0);
	
	// 64->128
	auto data_hi = _mm512_bsrli_epi128(data, 8);
	auto emptylenLo = _mm512_unpacklo_epi64(emptylen, emptylen);
	data = _mm512_mask_sllv_epi64(data, __mmask8(MASK_ALTERNATE), data, emptylenLo);
	data = _mm512_shrdv_epi64(data, data_hi, emptylenLo);
	
	// 128->256
	// get mask for >64 empty
	emptylen = _mm512_add_epi64(emptylen, _mm512_alignr_epi8(emptylen, emptylen, 8));
	emptylenLo = _mm512_permutex_epi64(emptylen, _MM_SHUFFLE(0,0,0,0));
	
	// TODO: these options don't work if emptylen == 128 (i.e. bits[128:255] need to go to [0:63]; perhaps that doesn't matter, as the only case we'll encounter 0 bits is at the end?
	
	/*
	// option1
	high = _mm512_maskz_mov_epi64(0b11001100, data); // TODO: probably better to do 2x masked permutes, then mask blend?  Uses an extra mask reg, but shorter dep chain; or maybe blend a permutexvar vector?
	high = _mm512_maskz_permutex_epi64(
		_mm512_cmple_epi64_mask(emptylenLo, _mm512_set1_epi64(64)),
		_mm512_permutex_epi64(data, _MM_SHUFFLE(0,1,3,2))
		_MM_SHUFFLE(0,3,2,1)
	);
	high = _mm512_shldv_epi64(
		high,
		_mm512_alignr_epi64(high, _mm512_setzero_si512(), 7),
		_mm512_sub_epi64(_mm512_set1_epi64(128), emptylenLo)
	);
	data = _mm512_or_si512(data, high);
	*/
	
	// option2
	auto empty1 = _mm512_cmpge_epi64_mask(emptylenLo, _mm512_set1_epi64(64));
	emptylenLo = _mm512_maskz_mov_epi64(MASK8(_kor)(empty1, MASK8(_cvtu32)(0b11101110)), emptylenLo);
	data = _mm512_mask_shldv_epi64(data, MASK8(_cvtu32)(0b00110011), _mm512_setzero_si512(), emptylenLo);
	data = _mm512_mask_permutex_epi64(data, empty1, data, _MM_SHUFFLE(1,3,2,0));
	data = _mm512_shrdv_epi64(
		data,
		_mm512_alignr_epi64(_mm512_setzero_si512(), data, 1), // this will pull in junk from high 256 to low 256 as it's unmasked; TODO: is this problematic - e.g. if we have 0 lengths, so when writing out, we could overwrite 0 data with junk?
		emptylenLo
	);
	
	/*
	// option3
	auto empty1 = _mm512_cmpge_epi64_mask(emptylenLo, _mm512_set1_epi64(64));
	data = _mm512_mask_shldv_epi64(
		data,
		MASK8(_cvtu32)(0b00110011),
		_mm512_alignr_epi64(data, _mm512_setzero_si512(), 7),
		emptylenLo
	);
	data = _mm512_mask_permutex_epi64(data, empty1, data, _MM_SHUFFLE(1,3,2,0));
	data = _mm512_shrdv_epi64(
		data,
		_mm512_alignr_epi64(_mm512_setzero_si512(), data, 1),
		emptylenLo
	);
	*/
	
	// 256->512
	emptylen = _mm512_add_epi64(emptylen, _mm512_permutex_epi64(emptylen, _MM_SHUFFLE(1,0,3,2)));
	emptylenLo = _mm512_broadcast_i64x4(_mm512_castsi512_si256(emptylen));
	
	data = _mm512_mask_shldv_epi64(
		data,
		MASK8(_cvtu32)(15),
		_mm512_alignr_epi64(data, _mm512_setzero_si512(), 7),
		emptylenLo
	);
	data = _mm512_maskz_compress_epi64(
		_mm512_cmplt_epi64_mask(emptylenLo, _mm512_set_epi64(512, 512, 512, 512, 64, 128, 192, 256)),
		data
	);
	data = _mm512_shrdv_epi64(
		data,
		_mm512_alignr_epi64(_mm512_setzero_si512(), data, 1),
		emptylenLo
	);
	
	// alternatively, extract top-256b and handle/store separately?
	
	
	
	total_bits = 512 - _mm_cvtsi128_si32(_mm_add_epi32(
		_mm512_castsi512_si128(emptylen), _mm512_extracti32x4_epi32(emptylen, 2)
	));
	assert(total_bits >= 0 && total_bits <= 512);
	return data;
}

static void huffman_encode(
	BitWriter& output,
	const HuffmanTree<286, 15>& huf_litlen, const HuffmanTree<30, 15>& huf_dist,
	const Lz77Data& lz77output
) {
	// load symbol lengths
	auto symlen0 = _mm512_load_si512(huf_litlen.lengths);
	auto symlen1 = _mm512_load_si512(huf_litlen.lengths + sizeof(__m512i));
	auto symlen2 = _mm512_load_si512(huf_litlen.lengths + sizeof(__m512i)*2);
	auto symlen3 = _mm512_load_si512(huf_litlen.lengths + sizeof(__m512i)*3);
	auto xsymlen = _mm512_inserti64x4(
		_mm512_castsi256_si512(_mm256_load_si256(reinterpret_cast<const __m256i*>(huf_litlen.lengths + sizeof(__m512i)*4))),
		_mm256_load_si256(reinterpret_cast<const __m256i*>(huf_dist.lengths)),
		1
	);
	
	// need to do a bit-rotate on lz77output.xbits_hi to make the main-loop code simpler
	// the number of bytes in lz77output.xbits_hi is expected to be low, so doesn't make sense to try doing this in-loop
	// it may make more sense to just overwrite lz77output.xbits_hi, but we'll retain its immutability for now
	alignas(64) uint8_t xbits_hi_tmp[LZ77DATA_XBITS_HI_SIZE];
	for(int i=0; i<lz77output.xbits_hi_len; i+=sizeof(__m512i)) {
		auto data = _mm512_load_si512(lz77output.xbits_hi + i);
		// max extra-bits length is 13 bits; since 7 is stored in the lo half, there can't be more than 6 bits set here
		assert(_mm512_test_epi8_mask(data, _mm512_set1_epi8(0b11000000)) == 0);
		
		// data = data >>> 1 (where >>> denotes bytewise bit-rotate)
		data = _mm512_gf2p8affine_epi64_epi8(data, _mm512_set1_epi64(0x0204081020408001ULL), 0);
		_mm512_store_si512(xbits_hi_tmp + i, data);
	}
	
	// compute Huffman codes
	alignas(64) uint16_t litlen_codes[286 +2];
	alignas(64) uint16_t dist_codes[30 +2];
	huf_litlen.CalcCodes(litlen_codes);
	huf_dist.CalcCodes(dist_codes);
	
	// TODO: if there's few symbols used, it may make sense to map these to a reduced alphabet to speed up lookups for those
	
	// separate low/high bytes of Huffman codes
	__m512i symlo0, symlo1, symlo2, symlo3, xsymlo;
	__m512i symhi0, symhi1, symhi2, symhi3, xsymhi;
	pack_bytes(
		_mm512_load_si512(litlen_codes),
		_mm512_load_si512(litlen_codes + 32),
		symlo0, symhi0
	);
	pack_bytes(
		_mm512_load_si512(litlen_codes + 64),
		_mm512_load_si512(litlen_codes + 96),
		symlo1, symhi1
	);
	pack_bytes(
		_mm512_load_si512(litlen_codes + 128),
		_mm512_load_si512(litlen_codes + 160),
		symlo2, symhi2
	);
	pack_bytes(
		_mm512_load_si512(litlen_codes + 192),
		_mm512_load_si512(litlen_codes + 224),
		symlo3, symhi3
	);
	pack_bytes(
		_mm512_load_si512(litlen_codes + 256),
		_mm512_load_si512(dist_codes),
		xsymlo, xsymhi
	);
	
	auto xbits_ptr = xbits_hi_tmp;
	auto last_data = _mm512_setzero_si512();
	for(unsigned srcpos=0; srcpos<lz77output.len; srcpos+=sizeof(__m512i)) { // overflow is not a problem, because the buffer is 0 padded
		auto data = _mm512_load_si512(lz77output.data + srcpos);
		auto m_data = _mm512_movepi8_mask(data);
		auto m_lendist = _cvtu64_mask64(lz77output.is_lendist[srcpos/sizeof(__m512i)]);
		auto m_xbits = _kandn_mask64(m_data, m_lendist);
		
		// shift data by a byte
		auto data_shift = _mm512_alignr_epi8(data, _mm512_alignr_epi64(data, last_data, 6), 15);
		last_data = data;
		
		// figure out literal symbol lengths
		auto code_bits = shuffle256(data, symlen0, symlen1, symlen2, symlen3);
		// add in length/distance symbol lengths
		code_bits = _mm512_mask_permutexvar_epi8(code_bits, m_lendist, data, xsymlen);
		// add in extra bit lengths
		code_bits = _mm512_mask_permutexvar_epi8(code_bits, m_xbits, data_shift, _mm512_set_epi32(
			// distance symbols
			0x0d0d, 0x0c0c0b0b, 0x0a0a0909, 0x08080707, 0x06060505, 0x04040303, 0x02020101, 0,
			// length symbols
			0x0005, 0x05050504, 0x04040403, 0x03030302, 0x02020201, 0x01010100, 0, 0
		));
		assert(_mm512_cmpgt_epu8_mask(code_bits, _mm512_set1_epi8(15)) == 0);
		
		// TODO: if all code_bits <= 8, consider skipping codehi
		// we could use the histogram to give an idea: if the likelihood of code_bits > 8 is low, add the check in
		
		// encode literal symbols
		auto codelo = shuffle256(data, symlo0, symlo1, symlo2, symlo3);
		auto codehi = shuffle256(data, symhi0, symhi1, symhi2, symhi3);
		// encode length/distance symbols
		codelo = _mm512_mask_permutexvar_epi8(codelo, m_lendist, data, xsymlo);
		codehi = _mm512_mask_permutexvar_epi8(codehi, m_lendist, data, xsymhi);
		assert(_mm512_test_epi8_mask(codehi, _mm512_set1_epi8(-128)) == 0); // max symbol length is 15 bits, so top bit should always be 0
		
		// blend in length/distance extra bits into codelo
		codelo = _mm512_mask_mov_epi8(codelo, m_xbits, data);
		// insert for codehi
		auto has_hi_xbits = _mm512_mask_test_epi8_mask(m_xbits, code_bits, _mm512_set1_epi8(8));
		codehi = _mm512_mask_expandloadu_epi8(codehi, has_hi_xbits, xbits_ptr);
		xbits_ptr += _mm_popcnt_u64(_cvtmask64_u64(has_hi_xbits));
		
		// codelo can only hold 7 xbits, so we need to move one bit from codehi to it
		const auto VALID_CODEHI_BITS = _mm512_set1_epi8(127);
		codelo = _mm512_ternarylogic_epi64(codelo, codehi, VALID_CODEHI_BITS, 0xf4); // A | (B & ~C)
		codehi = _mm512_and_si512(codehi, VALID_CODEHI_BITS);
		
		
		// combine codelo/hi together
		auto code0 = _mm512_unpacklo_epi8(codelo, codehi);
		auto code1 = _mm512_unpackhi_epi8(codelo, codehi);
		// instead of tracking code bit lengths, it's easier for us to track unoccupied bits, so compute relevant 'emptylen' for elements
		auto emptylen = _mm512_sub_epi8(_mm512_set1_epi8(16), code_bits);
		auto emptylen0 = _mm512_unpacklo_epi8(emptylen, _mm512_setzero_si512());
		auto emptylen1 = _mm512_unpackhi_epi8(emptylen, _mm512_setzero_si512());
		
		// concatenate 4x 16b elements into 64b elements
		code0 = huffman_bitjoin_16_64(code0, emptylen0);
		code1 = huffman_bitjoin_16_64(code1, emptylen1);
		
		// for compression to be effective, we generally expect <= 8 bits per symbol on average
		// if this is the case, we can take a shortcut
		emptylen = _mm512_sad_epu8(emptylen, _mm512_setzero_si512());
		assert(_mm512_cmpgt_epu64_mask(emptylen, _mm512_set1_epi64(128)) == 0); // TODO: this check is probably too loose
		assert(_mm512_cmplt_epu64_mask(emptylen, _mm512_set1_epi64(8)) == 0);
		//auto long_codes = _mm512_testn_epi8_mask(emptylen, _mm512_set1_epi64(192)); // emptylen < 64
		auto long_codes = _mm512_cmplt_epi8_mask(emptylen, _mm512_set1_epi64(64));
		if(!long_codes) {
			// since 8 symbols consume <=64b, we can combine everything into one vector
			
			// concatenate 2x64b elements into 1x64b
			emptylen0 = _mm512_unpacklo_epi64(emptylen0, emptylen1);
			emptylen0 = _mm512_sad_epu8(emptylen0, _mm512_setzero_si512());
			auto code0_ = _mm512_unpacklo_epi64(code0, code1);
			auto code1_ = _mm512_unpackhi_epi64(code0, code1);
			code0_ = _mm512_sllv_epi64(code0_, emptylen0);
			code0 = _mm512_shrdv_epi64(code0_, code1_, emptylen0);
			
			// 64b -> 512b
			int total_len;
			// TODO: this subtract 64 is probably unnecessary
			code0 = huffman_bitjoin_64_512(code0, _mm512_sub_epi64(emptylen, _mm512_set1_epi64(64)), total_len);
			output.ZeroWrite512(code0, total_len);
		} else {
			emptylen0 = _mm512_sad_epu8(emptylen0, _mm512_setzero_si512());
			emptylen1 = _mm512_sad_epu8(emptylen1, _mm512_setzero_si512());
			int total_len0, total_len1;
			
			// fix up permutation caused by earlier unpacklo/hi
			// TODO: see if avoiding this allows for optimisation opportunities
			const auto PERM0 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
			const auto PERM1 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
			auto code0_ = _mm512_permutex2var_epi64(code0, PERM0, code1);
			auto code1_ = _mm512_permutex2var_epi64(code0, PERM1, code1);
			auto emptylen0_ = _mm512_permutex2var_epi64(emptylen0, PERM0, emptylen1);
			auto emptylen1_ = _mm512_permutex2var_epi64(emptylen0, PERM1, emptylen1);
			
			// concatenate 64b -> 512b
			code0_ = huffman_bitjoin_64_512(code0_, emptylen0_, total_len0);
			code1_ = huffman_bitjoin_64_512(code1_, emptylen1_, total_len1);
			output.ZeroWrite505(code0_, total_len0);
			output.ZeroWrite505(code1_, total_len1);
		}
	}
}

// TODO: implement fixed huffman encoding

#endif
