#ifndef __SIMDFLATE_CHECKSUM_H
#define __SIMDFLATE_CHECKSUM_H

#include "common.hh"

// dummy checksum, used for deflate function which doesn't need one
class Checksum_None {
public:
	HEDLEY_ALWAYS_INLINE void update(__m512i) {}
	HEDLEY_ALWAYS_INLINE void update_partial(__m512i, int) {}
	void finalise(void*) {}
};

class Checksum_Adler32 {
	__m512i s1, s2, s1_sum;  // adler sums; s1_sum is the sum of s1 that will be multiplied by 64 and added to s2
	int rounds;  // number of 512-bit vectors processed; used to track when to perform reduction
	static constexpr int ADLER_MOD = 65521;
	// instead of dividing by ADLER_MOD, multiply the inverse; this is set up for 52-bit multiplication
	static constexpr uint64_t ADLER_INV52 = (1ULL << 52) / ADLER_MOD +1;
	
	// max rounds is bound by adl_s1_sum: largest n where ((n(n+1))/2)*(255*4) + 65520n < 2^32 - 1, which is [510n^2 + 66030n - 2^31-1 < 0], n=2837
	// adl_s1 max is 255n*4+1, and adl_s2 max is 255n*(64+63+62+61) (possibly a bit more for the partial vector), which are way below what adl_s1_sum reaches
	static constexpr int ADLER_MAX_ROUNDS = 2837;
	
	// our Adler computation gathers 32-bit sums, but IFMA52 prefers 64-bit sums; this function aggregates neighboring 32-bit sums into 64-bit sums
	// also merges s1_sum into s2
	void aggregate_adler() {
		// gather 64-bit sums into s1 and s2
		// s1/s2 are way below the 32-bit limit, so we can just do 32-bit adds, but s1_sum could hit the 32-bit limit, so do a 64-bit sum
		
		// double-check s1/s2 < 2^31
		assert(_mm512_test_epi32_mask(s1, _mm512_set1_epi32(0x80000000)) == 0);
		assert(_mm512_test_epi32_mask(s2, _mm512_set1_epi32(0x80000000)) == 0);
		
		const auto MASK_ALT = __mmask8(MASK_ALTERNATE);
		// using masked adds to zero the upper half of each 64-bit result
		s1 = _mm512_maskz_add_epi32(MASK_ALT, s1, _mm512_srli_epi64(s1, 32));
		s2 = _mm512_maskz_add_epi32(MASK_ALT, s2, _mm512_srli_epi64(s2, 32));
		s1_sum = _mm512_add_epi64(
			_mm512_maskz_mov_epi32(MASK_ALT, s1_sum),
			_mm512_srli_epi64(s1_sum, 32)
		);
		// combine s1_sum into s2
		s2 = _mm512_madd52lo_epu64( // is this a 'heavy' instruction?
			s2, s1_sum, _mm512_set1_epi64(64)
		);
	}
	// performs Adler reduction, i.e. computes % ADLER_MOD
	void compact_adler() {
		aggregate_adler();
		s1_sum = _mm512_setzero_si512();
		
		// compute mod 65521 for s1/s2
		// this works by multiplying the sum by 1/65521, discarding the quotient, multiplying the fractional remainder by 65521, to arrive at the result
		// in other words, it's the integer equivalent of n mod 65521 = frac(n/65521)*65521
		s1 = _mm512_madd52hi_epu64(
			_mm512_setzero_si512(),
			_mm512_madd52lo_epu64(_mm512_setzero_si512(), s1, _mm512_set1_epi64(ADLER_INV52)),
			_mm512_set1_epi64(ADLER_MOD)
		);
		s2 = _mm512_madd52hi_epu64(
			_mm512_setzero_si512(),
			_mm512_madd52lo_epu64(_mm512_setzero_si512(), s2, _mm512_set1_epi64(ADLER_INV52)),
			_mm512_set1_epi64(ADLER_MOD)
		);
		
		// verify s1/s2 < ADLER_MOD
		assert(_mm512_cmpge_epu32_mask(s1, _mm512_set1_epi32(ADLER_MOD)) == 0);
		assert(_mm512_cmpge_epu32_mask(s2, _mm512_set1_epi32(ADLER_MOD)) == 0);
		
		rounds = ADLER_MAX_ROUNDS;
	};
public:
	inline Checksum_Adler32() {
		s1 = ZEXT128_512(_mm_cvtsi32_si128(1));
		s2 = _mm512_setzero_si512();
		s1_sum = _mm512_setzero_si512();
		rounds = ADLER_MAX_ROUNDS;
	}
	HEDLEY_ALWAYS_INLINE void update(__m512i data) {
		s1_sum = _mm512_add_epi32(s1_sum, s1);
		s1 = _mm512_dpbusd_epi32(s1, data, _mm512_set1_epi8(1));
		s2 = _mm512_dpbusd_epi32(s2, data, VEC512_8(64-_x));
		if(HEDLEY_UNLIKELY(--rounds == 0))
			compact_adler();
	}
	HEDLEY_ALWAYS_INLINE void update_partial(__m512i data, int len) {
		// invalid data items are assumed to be 0
		assert((_cvtmask64_u64(_mm512_test_epi8_mask(data, data)) >> len) == 0);
		assert(len > 0 && len <= int(sizeof(__m512i)));
		
		// sum directly to s2 instead of s1_sum, as it uses a different coefficient
		s2 = _mm512_add_epi32(s2, _mm512_mullo_epi32(s1, _mm512_set1_epi32(len))); // is _mm512_mullo_epi32 is a 'heavy' instruction?
		s1 = _mm512_dpbusd_epi32(s1, data, _mm512_set1_epi8(1));
		s2 = _mm512_dpbusd_epi32(s2, data, _mm512_add_epi8(VEC512_8(64-_x), _mm512_set1_epi8(len - sizeof(__m512i))));
		compact_adler(); // can't be bothered trying to track s2, so just always compact; update_partial is only called once anyway
	}
	void finalise(void* dest) {
		aggregate_adler();
		// combine s1/s2 into one vector
		s1 = _mm512_add_epi64(
			_mm512_unpacklo_epi64(s1, s2),
			_mm512_unpackhi_epi64(s1, s2)
		);
		// reduce sum to 128b vector
		auto tmp256 = _mm256_add_epi64(_mm512_castsi512_si256(s1), _mm512_extracti64x4_epi64(s1, 1));
		auto adler = _mm_add_epi64(_mm256_castsi256_si128(tmp256), _mm256_extracti128_si256(tmp256, 1));
		
		// mod 65521
		adler = _mm_madd52hi_epu64(
			_mm_setzero_si128(),
			_mm_madd52lo_epu64(_mm_setzero_si128(), adler, _mm512_castsi512_si128(_mm512_set1_epi64(ADLER_INV52))),
			_mm512_castsi512_si128(_mm512_set1_epi64(ADLER_MOD))
		);
		// shuffle into position + endian swap
		adler = _mm_shuffle_epi8(adler, _mm_set_epi32(-1, -1, -1, 0x00010809));
		
		// write out the checksum
		uint32_t final_adler = _mm_cvtsi128_si32(adler);
		memcpy(dest, &final_adler, sizeof(final_adler));
	}
};

class Checksum_Crc32 {
	__m512i crc;
public:
	inline Checksum_Crc32() {
		crc = ZEXT128_512(_mm_cvtsi32_si128(0x9db42487));
	}
	HEDLEY_ALWAYS_INLINE void update(__m512i data) {
		const auto CRC_FOLD = _mm512_set4_epi32(1, 0x54442bd4, 1, 0xc6e41596);
		crc = _mm512_ternarylogic_epi64(
			_mm512_clmulepi64_epi128(crc, CRC_FOLD, 0x01),
			_mm512_clmulepi64_epi128(crc, CRC_FOLD, 0x10),
			data, 0x96 // 0x96 = A^B^C
		);
	}
	HEDLEY_ALWAYS_INLINE void update_partial(__m512i data, int len) {
		assert(len > 0 && len <= int(sizeof(__m512i)));
		// shift things around into position
		auto idx_shift = _mm512_add_epi8(_mm512_set1_epi8(len), VEC512_8(_x));
		auto crc_shift = _mm512_maskz_permutexvar_epi8(
			_cvtu64_mask64(0xffffffffffffffffULL << (64-len)),
			idx_shift, crc
		);
		crc = _mm512_permutex2var_epi8(crc, idx_shift, data);
		
		const auto CRC_FOLD = _mm512_set4_epi32(1, 0x54442bd4, 1, 0xc6e41596);
		crc = _mm512_ternarylogic_epi64(crc,
			_mm512_clmulepi64_epi128(crc_shift, CRC_FOLD, 0x01),
			_mm512_clmulepi64_epi128(crc_shift, CRC_FOLD, 0x10),
			0x96
		);
	}
	void finalise(void* dest) {
		// 512b -> 128b
		auto crc_const512 = _mm512_set_epi32(
			0, 0, 0, 0,
			1, 0x751997d0, 0, 0xccaa009e,
			0, 0xf1da05aa, 1, 0x5a546366,
			0, 0x3db1ecdc, 1, 0x74359406
		);
		crc = _mm512_ternarylogic_epi64(
			_mm512_clmulepi64_epi128(crc, crc_const512, 0x01),
			_mm512_clmulepi64_epi128(crc, crc_const512, 0x10),
			ZEXT128_512(_mm512_extracti32x4_epi32(crc, 3)),
			0x96
		);
		auto crc_temp = _mm_ternarylogic_epi64(
			_mm512_castsi512_si128(crc),
			_mm256_extracti128_si256(_mm512_castsi512_si256(crc), 1),
			_mm512_extracti32x4_epi32(crc, 2),
			0x96
		);
		
		// 128b -> 64b
		auto crc_const = _mm_set_epi32(1, 0x63cd6124, 0, 0xccaa009e);
		crc_temp = _mm_xor_si128(
			_mm_clmulepi64_si128(crc_temp, crc_const, 0),
			_mm_srli_si128(crc_temp, 8)
		);
		
		// 64b -> 32b
		crc_temp = _mm_ternarylogic_epi64(
			_mm_clmulepi64_si128(_mm_slli_epi64(crc_temp, 32), crc_const, 0x10),
			crc_temp,
			_mm_set_epi32(-1,-1,-1,0),
			0x28 // (A ^ (B & C))
		);
		
		// final reduction + invert bits
		crc_const = _mm_set_epi32(1, 0xdb710640, 0, 0xf7011641);
		crc_temp = _mm_ternarylogic_epi64(
			_mm_clmulepi64_si128(
				_mm_clmulepi64_si128(crc_temp, crc_const, 0),
				crc_const, 0x10
			),
			crc_temp, crc_temp,
			0xC3 // (A ^ ~B)
		);
		
		// write out checksum
		uint32_t final_crc = _mm_extract_epi32(crc_temp, 2);
		memcpy(dest, &final_crc, sizeof(final_crc));
	}
};

#endif
