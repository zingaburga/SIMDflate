#ifndef __SIMDFLATE_HUFFMANENC_H
#define __SIMDFLATE_HUFFMANENC_H

#include "common.hh"
#include "bitwriter.hh"
#include "huffmantree.hh"
#include "lz77data.hh"


// the number of extra bits for corresponding length/distance symbol
static HEDLEY_ALWAYS_INLINE __m512i huffman_lendist_xbits_len() {
	return _mm512_set_epi32(
		// distance symbols
		0x0d0d, 0x0c0c0b0b, 0x0a0a0909, 0x08080707, 0x06060505, 0x04040303, 0x02020101, 0,
		// length symbols
		0x0005, 0x05050504, 0x04040403, 0x03030302, 0x02020201, 0x01010100, 0, 0
	);
}

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

static void huffman_join_write(BitWriter& output, __m512i codelen, __m512i codelo, __m512i codehi) {
	// combine codelo/hi together
	auto code0 = _mm512_unpacklo_epi8(codelo, codehi);
	auto code1 = _mm512_unpackhi_epi8(codelo, codehi);
	// instead of tracking code bit lengths, it's easier for us to track unoccupied bits, so compute relevant 'emptylen' for elements
	auto emptylen = _mm512_sub_epi8(_mm512_set1_epi8(16), codelen);
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

// dynamic Huffman encode
static void huffman_dyn_encode(
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
	
#if WINDOW_ORDER > 9
	auto xbits_ptr = lz77output.xbits_hi;
#endif
	
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
		auto codelen = shuffle256(data, symlen0, symlen1, symlen2, symlen3);
		// add in length/distance symbol lengths
		codelen = _mm512_mask_permutexvar_epi8(codelen, m_lendist, data, xsymlen);
		// add in extra bit lengths
		codelen = _mm512_mask_permutexvar_epi8(codelen, m_xbits, data_shift, huffman_lendist_xbits_len());
		assert(_mm512_cmpgt_epu8_mask(codelen, _mm512_set1_epi8(15)) == 0);
		
		// TODO: if all codelen <= 8, consider skipping codehi
		// we could use the histogram to give an idea: if the likelihood of codelen > 8 is low, add the check in
		
		// encode literal symbols
		auto codelo = shuffle256(data, symlo0, symlo1, symlo2, symlo3);
		auto codehi = shuffle256(data, symhi0, symhi1, symhi2, symhi3);
		// encode length/distance symbols
		codelo = _mm512_mask_permutexvar_epi8(codelo, m_lendist, data, xsymlo);
		codehi = _mm512_mask_permutexvar_epi8(codehi, m_lendist, data, xsymhi);
		assert(_mm512_test_epi8_mask(codehi, _mm512_set1_epi8(-128)) == 0); // max symbol length is 15 bits, so top bit should always be 0
		
		// blend in length/distance extra bits into codelo
		codelo = _mm512_mask_mov_epi8(codelo, m_xbits, data);
#if WINDOW_ORDER > 9
		// insert high extra bits into codehi
		auto has_hi_xbits = _mm512_mask_test_epi8_mask(m_xbits, codelen, _mm512_set1_epi8(8));
		codehi = _mm512_mask_expandloadu_epi8(codehi, has_hi_xbits, xbits_ptr);
		xbits_ptr += _mm_popcnt_u64(_cvtmask64_u64(has_hi_xbits));
		
		// codelo can only hold 7 xbits, so we need to move one bit from codehi to it
		const auto VALID_CODEHI_BITS = _mm512_set1_epi8(127);
		codelo = _mm512_ternarylogic_epi64(codelo, codehi, VALID_CODEHI_BITS, 0xf4); // A | (B & ~C)
		codehi = _mm512_and_si512(codehi, VALID_CODEHI_BITS);
#endif
		
		huffman_join_write(output, codelen, codelo, codehi);
	}
}


static void huffman_fixed_encode(BitWriter& output, const Lz77Data& lz77output) {
#if WINDOW_ORDER > 9
	auto xbits_ptr = lz77output.xbits_hi;
#endif
	
	auto last_data = _mm512_setzero_si512();
	for(unsigned srcpos=0; srcpos<lz77output.len; srcpos+=sizeof(__m512i)) {
		auto data = _mm512_load_si512(lz77output.data + srcpos);
		auto m_data = _mm512_movepi8_mask(data);
		auto m_lendist = _cvtu64_mask64(lz77output.is_lendist[srcpos/sizeof(__m512i)]);
		auto m_xbits = _kandn_mask64(m_data, m_lendist);
		
		// shift data by a byte
		auto data_shift = _mm512_alignr_epi8(data, _mm512_alignr_epi64(data, last_data, 6), 15);
		last_data = data;
		
		// figure out literal symbol lengths
		// since length boundaries are multiples of 8, we can do a lookup on that specifically
		auto len_lut = _mm512_srli_epi16(data, 3);
		len_lut = _mm512_ternarylogic_epi64(
			len_lut, _mm512_movm_epi8(m_lendist), _mm512_set1_epi8(-32), 0xd8 // (A&~C) | (B&C)
		);
		// ...alternatively, affine + mask-add
		// or do a >=144 compare, blend 8/9, mask-permute
		
		auto symlen = _mm512_permutexvar_epi8(len_lut, _mm512_set_epi8(
			0,0,0,0,0,0,0,0,
			5,5,5,5,                            // 288-319 (distance 0-31)
			8,                                  // 280-287
			7,7,7,                              // 256-279
			0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,    // reserved for extra bits
			9,9,9,9,9,9,9,9,9,9,9,9,9,9,        // 144-255
			8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8 // 0-143
		));
		// add in extra bit lengths
		auto codelen = _mm512_mask_permutexvar_epi8(symlen, m_xbits, data_shift, huffman_lendist_xbits_len());
		assert(_mm512_cmpgt_epu8_mask(codelen, _mm512_set1_epi8(13)) == 0);
		
		
		// encode literal symbols
		auto codebase = _mm512_permutexvar_epi8(len_lut, _mm512_set_epi8(
			0,0,0,0,0,0,0,0,
			96,96,96,96,                        // 288-319 (-160)
			32,                                 // 280-287 (0b1100'0000 - 160 = 32)
			-128,-128,-128,                     // 256-279 (-128)
			0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,    // reserved for extra bits
			0,0,0,0,0,0,0,0,0,0,0,0,0,0,        // 144-255 (0b1001'0000 - 144 = 0; top bit is always 1, so ignored)
			48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48,48  // 0-143 (0b0011'0000)
		));
		auto codelo = _mm512_add_epi8(data, codebase);
		
		// symbols with 5/7-bit codes need to get shifted, before being reversed; this is so that the reversed representation is properly aligned
		codelo = _mm512_gf2p8mul_epi8(codelo, _mm512_gf2p8affine_epi64_epi8(
			symlen, _mm512_set1_epi64(0x04020006'00000000), 1  // map 5->8, 7->2, else->1
		));
		
		// for nine bit symbols, the least significant bit will become the most significant after bit-reversal
		// so move this bit into codehi
		auto nine_bit_syms = _mm512_cmpgt_epi8_mask(symlen, _mm512_set1_epi8(8));
		//auto codehi = _mm512_and_si512(_mm512_maskz_mov_epi8(nine_bit_syms, codelo), _mm512_set1_epi8(1));
		auto codehi = _mm512_and_si512(codelo, _mm512_subs_epu8(symlen, _mm512_set1_epi8(8)));
		//auto codehi = _mm512_maskz_mov_epi8(_mm512_mask_test_epi8_mask(nine_bit_syms, codelo, _mm512_set1_epi8(1)), _mm512_set1_epi8(1));
		// shift codelo right by 1 (getting rid of the bit we moved to codehi), with a 1 being shifted in  [i.e. (codelo >> 1) + 128]
		codelo = _mm512_mask_avg_epu8(codelo, nine_bit_syms, codelo, _mm512_set1_epi8(-1));
		
		// reverse bits
		codelo = _mm512_mask_gf2p8affine_epi64_epi8(
			codelo, _knot_mask64(m_xbits),
			codelo, _mm512_set1_epi64(0x8040201008040201ULL), 0
		);
		
#if WINDOW_ORDER > 9
		// insert high extra bits into codehi
		auto has_hi_xbits = _mm512_mask_test_epi8_mask(m_xbits, codelen, _mm512_set1_epi8(8));
		codehi = _mm512_mask_expandloadu_epi8(codehi, has_hi_xbits, xbits_ptr);
		xbits_ptr += _mm_popcnt_u64(_cvtmask64_u64(has_hi_xbits));
		
		// codelo can only hold 7 xbits, so we need to move one bit from codehi to it
		const auto VALID_CODEHI_BITS = _mm512_set1_epi8(127);
		codelo = _mm512_ternarylogic_epi64(codelo, codehi, VALID_CODEHI_BITS, 0xf4); // A | (B & ~C)
		codehi = _mm512_and_si512(codehi, VALID_CODEHI_BITS);
#endif
		
		huffman_join_write(output, codelen, codelo, codehi);
	}
}

// Encode <= 64 bytes using fixed Huffman
static void huffman_fixed_encode_literals(BitWriter& output, __m512i data, __mmask64 valid_mask) {
	auto nine_bit_syms = _mm512_cmpge_epu8_mask(data, _mm512_set1_epi8(int8_t(144)));
	auto eight = _mm512_maskz_set1_epi8(valid_mask, 8);
	auto codelen = _mm512_mask_blend_epi8(
		nine_bit_syms, eight, _mm512_set1_epi8(9)
	);
	auto codelo = _mm512_mask_blend_epi8(
		nine_bit_syms,
		_mm512_maskz_add_epi8(valid_mask, data, _mm512_set1_epi8(48)),
		_mm512_avg_epu8(data, _mm512_set1_epi8(-1))
	);
	auto codehi = _mm512_ternarylogic_epi64(data, codelen, eight, 0x60); // A & (B ^ C)
	
	// reverse bits
	codelo = _mm512_gf2p8affine_epi64_epi8(
		codelo, _mm512_set1_epi64(0x8040201008040201ULL), 0
	);
	
	huffman_join_write(output, codelen, codelo, codehi);
}

#endif
