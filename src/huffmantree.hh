#ifndef __SIMDFLATE_HUFFMANTREE_H
#define __SIMDFLATE_HUFFMANTREE_H

#include "common.hh"

// 256 byte entry permute on data, with lookup table in tbl0-3
static HEDLEY_ALWAYS_INLINE __m512i shuffle256(__m512i data, __m512i tbl0, __m512i tbl1, __m512i tbl2, __m512i tbl3) {
	return _mm512_mask_blend_epi8(
		_mm512_movepi8_mask(data),
		_mm512_permutex2var_epi8(tbl0, data, tbl1),
		_mm512_permutex2var_epi8(tbl2, data, tbl3)
	);
}

// split 16-bit elements into low/high 8-bit elements
// takes 16-bit inputs in in0/1 and puts the low 8 bits in outl, and the high 8 bits in outh
static HEDLEY_ALWAYS_INLINE void pack_bytes(__m512i in0, __m512i in1, __m512i& outl, __m512i& outh) {
	const auto PACK_PERM = VEC512_8(((_x&31)<<1) | ((_x&32) >> 5));
	in0 = _mm512_permutexvar_epi8(PACK_PERM, in0);
	in1 = _mm512_permutexvar_epi8(PACK_PERM, in1);
	outl = _mm512_inserti64x4(in0, _mm512_castsi512_si256(in1), 1);
	outh = _mm512_shuffle_i64x2(in0, in1, _MM_SHUFFLE(3,2,3,2));
	
	// alternative: could be better due to less cross-lane permutes, but has constants to load
	/*
	const auto PACK_SHUF = _mm512_set4_epi32(0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200);
	in0 = _mm512_shuffle_epi8(in0, PACK_SHUF);
	in1 = _mm512_shuffle_epi8(in1, PACK_SHUF);
	outl = _mm512_permutex2var_epi64(in0, _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0), in1);
	outh = _mm512_permutex2var_epi64(in0, _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1), in1);
	*/
}



// pre-computed lookup table for {0, 1/1, 1/3, 1/7, 1/15 ...}
constexpr uint32_t divide_gen(unsigned n) {
	if(n == 0) return 0;
	if(n == 1) return 1<<31;
	return (1U<<31) / ((1<<n) -1) +1;
}
static constexpr auto TO_MAX_DIVIDE = lut<16>(divide_gen);



template<int size, int max_length>
class HuffmanTree {
	
	/// Histogram helpers
	
	//  for small alphabet histogramming, it can be done by testing for each letter of the alphabet, and counting the matches
	static HEDLEY_ALWAYS_INLINE __m128i hist_match_pair(__m512i data, int n) {
		auto match0 = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8(n)));
		auto match1 = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8(n+1)));
		return _mm_insert_epi64(_mm_cvtsi64_si128(match0), match1, 1);
	}
	// like above function, but if `data` has <= 32 elements, we can do two matches at once with 512-bit vectors
	static HEDLEY_ALWAYS_INLINE __m128i hist_match_pair_x2(__m512i data, int n) {
		auto match0 = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(data, VEC512_8(_x/32+n)));
		auto match1 = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(data, VEC512_8(_x/32+n+2)));
		return _mm_insert_epi64(_mm_cvtsi64_si128(match0), match1, 1);
	}
	
	/// Sorting helpers
	
	// moves masked bytes to the low-value side of the vector (like compress), and unmasked bytes to the high-value side
	// original order of bytes is otherwise preserved
	static HEDLEY_ALWAYS_INLINE __m512i splice_i8(__m512i data, __mmask64 lo_elems) {
		auto hi = _mm512_maskz_compress_epi8(_knot_mask64(lo_elems), data);
		/*
		auto lo = _mm512_maskz_compress_epi8(lo_elems, data);
		uint64_t expmask = ~0ULL << _mm_popcnt_u64(_cvtmask64_u64(lo_elems));
		return _mm512_mask_expand_epi8(lo, _cvtu64_mask64(expmask), hi);
		*/
		auto shift_idx = _mm512_set1_epi8(_mm_popcnt_u64(_cvtmask64_u64(lo_elems)));
		shift_idx = _mm512_sub_epi8(VEC512_8(_x), shift_idx);
		hi = _mm512_permutexvar_epi8(shift_idx, hi);
		return _mm512_mask_compress_epi8(hi, lo_elems, data);
	}
	static HEDLEY_ALWAYS_INLINE __m512i splice_i16(__m512i data, __mmask32 lo_elems) {
		auto hi = _mm512_maskz_compress_epi16(_knot_mask32(lo_elems), data);
		/*
		auto lo = _mm512_maskz_compress_epi16(lo_elems, data);
		uint32_t expmask = ~0U << _mm_popcnt_u32(_cvtmask32_u32(lo_elems));
		return _mm512_mask_expand_epi16(lo, _cvtu32_mask32(expmask), hi);
		*/
		auto shift_idx = _mm512_set1_epi16(_mm_popcnt_u32(_cvtmask32_u32(lo_elems)));
		shift_idx = _mm512_sub_epi16(VEC512_16(_x), shift_idx);
		hi = _mm512_permutexvar_epi16(shift_idx, hi);
		return _mm512_mask_compress_epi16(hi, lo_elems, data);
	}
	// faster versions of above, for when we know half the elements go on either side
	static HEDLEY_ALWAYS_INLINE __m512i splicehalf_i8(__m512i data, __mmask64 lo_elems) {
		assert(_mm_popcnt_u64(_cvtmask64_u64(lo_elems)) == 32);
		auto lo = _mm512_maskz_compress_epi8(lo_elems, data);
		auto hi = _mm512_maskz_compress_epi8(_knot_mask64(lo_elems), data);
		return _mm512_inserti64x4(lo, _mm512_castsi512_si256(hi), 1);
	}
	static HEDLEY_ALWAYS_INLINE __m512i splicehalf_i16(__m512i data, __mmask32 lo_elems) {
		assert(_mm_popcnt_u32(_cvtmask32_u32(lo_elems)) == 16);
		auto lo = _mm512_maskz_compress_epi16(lo_elems, data);
		auto hi = _mm512_maskz_compress_epi16(_knot_mask32(lo_elems), data);
		return _mm512_inserti64x4(lo, _mm512_castsi512_si256(hi), 1);
	}
	static HEDLEY_ALWAYS_INLINE __m512i sort_hi32(__m512i si) {
		auto test = _mm512_set1_epi16(512);
		// skip first bit - sort later
		si = splicehalf_i16(si, _mm512_testn_epi16_mask(si, test));
		test = _mm512_add_epi16(test, test);
		si = splicehalf_i16(si, _mm512_testn_epi16_mask(si, test));
		test = _mm512_add_epi16(test, test);
		si = splicehalf_i16(si, _mm512_testn_epi16_mask(si, test));
		test = _mm512_add_epi16(test, test);
		si = splicehalf_i16(si, _mm512_testn_epi16_mask(si, test));
		
		/*
		// swap bottom 2 bits using a bitonic style sort
		auto rot = _mm512_rol_epi64(si, 32);
		auto min = _mm512_min_epu16(si, rot);
		si = _mm512_mask_max_epu16(min, _cvtu32_mask32(0xcccccccc), si, rot);
		
		rot = _mm512_shuffle_epi8(si, _mm512_set4_epi32(
			0x09080b0a, 0x0d0c0f0e, 0x01000302, 0x05040706
		));
		min = _mm512_min_epu16(si, rot);
		si = _mm512_mask_max_epu16(min, _cvtu32_mask32(0xcccccccc), si, rot);
		*/
		// sort first bit
		auto swap = _mm512_test_epi32_mask(si, _mm512_set1_epi32(256));
		assert(_mm_popcnt_u32(_cvtmask32_u32(swap)) == 16);
		si = _mm512_mask_rol_epi32(si, swap, si, 16);
		return si;
	}
	
	/// Bit writing helpers
	static HEDLEY_ALWAYS_INLINE void bit_write64(void* output, uint64_t data, int len, uint_fast8_t& byte, int bits) {
		uint64_t out_data = (data << bits) | byte;
		memcpy(output, &out_data, sizeof(out_data));
		auto len_shifted = len + bits;
		if(HEDLEY_UNLIKELY(len_shifted > 64)) { // TODO: might not be that unlikely (e.g. if lots of 0s)
			byte = data >> (64 - bits);
		} else if(len_shifted == 64) {
			byte = 0;
		} else {
			byte = out_data >> (len_shifted & ~7);
		}
		
		/* TODO: this code doesn't work
		auto len_total = len + bits;
		if(len_total >= 64) {
			uint32_t out_data0 = (data << bits) | byte;
			memcpy(output, &out_data0, sizeof(out_data0));
			data >>= 32 - bits;
			len -= 32 - bits;
			memcpy(static_cast<char*>(output) + 4, &data, sizeof(data));
			byte = data >> (len & ~7);
		} else {
			uint64_t out_data = (data << bits) | byte;
			memcpy(output, &out_data, sizeof(out_data));
			byte = out_data >> (len_total & ~7);
		}
		*/
	}
	static HEDLEY_ALWAYS_INLINE void bit_write32flush(void* output, uint32_t data, int len, uint_fast8_t byte, int bits) {
		uint64_t out_data = (uint64_t(data) << bits) | byte;
		auto len_shifted = len + bits;
		if(HEDLEY_UNLIKELY(len_shifted > 32)) {
			memcpy(output, &out_data, 5);
		} else {
			memcpy(output, &out_data, 4);
		}
	}
	
	// inverse of pack_bytes
	// takes low/high bytes from inl and inh, returns joined 16-bit elements in out0 and out1
	static HEDLEY_ALWAYS_INLINE void unpack_bytes(__m512i inl, __m512i inh, __m512i& out0, __m512i& out1) {
		const auto SEP_PERM = VEC512_8((_x&7) | ((_x&8) << 2) | ((_x&48) >> 1));
		inl = _mm512_permutexvar_epi8(SEP_PERM, inl);
		inh = _mm512_permutexvar_epi8(SEP_PERM, inh);
		out0 = _mm512_unpacklo_epi8(inl, inh);
		out1 = _mm512_unpackhi_epi8(inl, inh);
	}
	
	// lookup over 1-4x 64-byte vectors
	template<int num_sym>
	static HEDLEY_ALWAYS_INLINE __m512i ShuffleIdx(__m512i data, __m512i idx0, __m512i idx1, __m512i idx2, __m512i idx3) {
		if(num_sym == 64)
			return _mm512_permutexvar_epi8(data, idx0);
		else if(num_sym == 128)
			return _mm512_permutex2var_epi8(idx0, data, idx1);
		else if(num_sym == 192)
			return _mm512_mask_permutexvar_epi8(
				_mm512_permutex2var_epi8(idx0, data, idx1),
				_mm512_movepi8_mask(data),
				data, idx2
			);
		return shuffle256(data, idx0, idx1, idx2, idx3);
	}
	
	template<int num_sym, typename IdxType>
	static void SortHistLitlen(const uint16_t* sym_counts, IdxType* idx_sorted, uint16_t* hist_sorted) {
		HEDLEY_STATIC_ASSERT(num_sym == 64 || num_sym == 128 || num_sym == 192 || num_sym == 256 || num_sym == 288, "Invalid number of elements to sort");
		HEDLEY_STATIC_ASSERT(sizeof(IdxType) == 1+(num_sym==288), "Invalid type for number of elements");
		
		// split lo/hi counts
		__m512i sc0l, sc0h, sc1l, sc1h, sc2l, sc2h, sc3l, sc3h;
		__m256i sc4l, sc4h;
		// shut up compiler warnings
		sc1l = sc1h = sc2l = sc2h = sc3l = sc3h = _mm512_undefined_epi32();
		sc4l = sc4h = _mm256_undefined_si256();
		pack_bytes(
			_mm512_loadu_si512(sym_counts),
			_mm512_loadu_si512(sym_counts + 32),
			sc0l, sc0h
		);
		if(num_sym > 64) {
			pack_bytes(
				_mm512_loadu_si512(sym_counts + 64),
				_mm512_loadu_si512(sym_counts + 96),
				sc1l, sc1h
			);
		}
		if(num_sym > 128) {
			pack_bytes(
				_mm512_loadu_si512(sym_counts + 128),
				_mm512_loadu_si512(sym_counts + 160),
				sc2l, sc2h
			);
		}
		if(num_sym > 192) {
			pack_bytes(
				_mm512_loadu_si512(sym_counts + 192),
				_mm512_loadu_si512(sym_counts + 224),
				sc3l, sc3h
			);
		}
		const auto LAST_MASK = _cvtu32_mask32((1 << 30) -1);
		if(num_sym > 256) {
			auto sc4t = _mm512_permutexvar_epi8(VEC512_8(((_x&31)<<1) | ((_x&32) >> 5)), _mm512_loadu_si512(sym_counts + 256));
			sc4l = _mm512_castsi512_si256(sc4t);
			sc4h = _mm512_extracti64x4_epi64(sc4t, 1);
		}
		
		alignas(64) uint8_t idx_store[num_sym*2];
		uint8_t islen_store[num_sym*2/8] = {}; // only used for num_sym > 256
		int lo_pos = 0, hi_pos = num_sym;
		auto test = _mm512_set1_epi8(1);
		
		// split first round
		//auto idx = VEC512_8((_x&7) | ((_x&8) << 2) | ((_x&48) >> 1));
		auto idx = VEC512_8(_x);
		auto split_round0 = [&](__m512i elems) {
			auto lo_elems = _mm512_testn_epi8_mask(elems, test);
			int loel_cnt = _mm_popcnt_u64(_cvtmask64_u64(lo_elems));
			compress_store_512_8(idx_store + lo_pos, lo_elems, idx);
			lo_pos += loel_cnt;
			if(num_sym <= 64) {
				compress_store_512_8(idx_store + loel_cnt, _knot_mask64(lo_elems), idx);
			} else {
				compress_store_512_8(idx_store + hi_pos, _knot_mask64(lo_elems), idx);
				hi_pos += 64 - loel_cnt;
			}
			idx = _mm512_add_epi8(idx, _mm512_set1_epi8(64));
		};
		split_round0(sc0l);
		if(num_sym > 64)  split_round0(sc1l);
		if(num_sym > 128) split_round0(sc2l);
		if(num_sym > 192) split_round0(sc3l);
		if(num_sym > 256) {
			auto lo_elems = _mm256_mask_testn_epi8_mask(LAST_MASK, sc4l, _mm512_castsi512_si256(test));
			int loel_cnt = _mm_popcnt_u32(_cvtmask32_u32(lo_elems));
			compress_store_256_8(idx_store + lo_pos, lo_elems, _mm512_castsi512_si256(idx));
			compress_store_256_8(idx_store + hi_pos, _knot_mask32(lo_elems), _mm512_castsi512_si256(idx));
			uint64_t lo_islen = _bzhi_u32(-1, loel_cnt) << (lo_pos & 7);
			uint64_t hi_islen = _bzhi_u32(-1, 30-loel_cnt) << (hi_pos & 7);
			memcpy(islen_store + (lo_pos>>3), &lo_islen, sizeof(lo_islen));
			memcpy(islen_store + (hi_pos>>3), &hi_islen, sizeof(hi_islen));
			lo_pos += loel_cnt;
			hi_pos += 30 - loel_cnt;
		}
		
		
		auto shuffle_idx = [](__m512i idx, __m512i sc0, __m512i sc1, __m512i sc2, __m512i sc3, uint64_t islen, __m256i sc4) -> __m512i {
			if(num_sym > 256) {
				// TODO: consider storing shuffled elements instead of re-looking them up
				auto elems = shuffle256(idx, sc0, sc1, sc2, sc3);
				elems = _mm512_mask_permutexvar_epi8(elems, _cvtu64_mask64(islen), idx, _mm512_castsi256_si512(sc4));
				return elems;
			} else
				return ShuffleIdx<num_sym>(idx, sc0, sc1, sc2, sc3);
		};
		
		auto merge_lohi_store = [&]() {
			if(num_sym > 256) {
				_mm256_storeu_si256(reinterpret_cast<__m256i*>(idx_store + lo_pos), _mm256_load_si256(reinterpret_cast<const __m256i*>(idx_store + num_sym)));
				for(int j=32; j<num_sym; j+=64)
					_mm512_storeu_si512(idx_store + lo_pos + j, _mm512_load_si512(idx_store + num_sym + j));
				auto islen_hi = _mm512_maskz_loadu_epi32(_cvtu32_mask16((1<<(num_sym/32)) -1), islen_store + num_sym/8);
				islen_hi = shift_bits_into_vec(islen_hi, islen_store[lo_pos>>3], lo_pos & 7);
				_mm512_mask_storeu_epi8(islen_store + (lo_pos>>3), _cvtu64_mask64((1ULL << (num_sym/8)) -1), islen_hi);
			} else if(num_sym > 64) {
				for(int j=0; j<num_sym; j+=64)
					_mm512_storeu_si512(idx_store + lo_pos + j, _mm512_load_si512(idx_store + num_sym + j));
			}
		};
		auto sort_round = [&](__m512i sc0, __m512i sc1, __m512i sc2, __m512i sc3, __m256i sc4) {
			merge_lohi_store();
			
			lo_pos = 0; hi_pos = num_sym;
			uint_fast8_t lo_byte = 0, hi_byte = 0;
			auto split_round = [&](__m512i idx, uint64_t islen) {
				auto elems = shuffle_idx(idx, sc0, sc1, sc2, sc3, islen, sc4);
				auto lo_elems = _mm512_testn_epi8_mask(elems, test);
				uint64_t i_lo_elems = _cvtmask64_u64(lo_elems);
				int loel_cnt = _mm_popcnt_u64(i_lo_elems);
				compress_store_512_8(idx_store + lo_pos, lo_elems, idx);
				if(num_sym <= 64)
					compress_store_512_8(idx_store + loel_cnt, _knot_mask64(lo_elems), idx);
				else
					compress_store_512_8(idx_store + hi_pos, _knot_mask64(lo_elems), idx);
				if(num_sym > 256) {
					uint64_t lo_islen = _pext_u64(islen, i_lo_elems);
					uint64_t hi_islen = _pext_u64(islen, ~i_lo_elems);
					// TODO: vectorize these two lines?
					bit_write64(islen_store + (lo_pos>>3), lo_islen, loel_cnt, lo_byte, lo_pos&7);
					bit_write64(islen_store + (hi_pos>>3), hi_islen, 64 - loel_cnt, hi_byte, hi_pos&7);
				}
				lo_pos += loel_cnt;
				hi_pos += 64 - loel_cnt;
			};
			auto* islen_store64 = reinterpret_cast<const uint64_t*>(islen_store);
			split_round(_mm512_load_si512(idx_store), islen_store64[0]);
			if(num_sym > 64)  split_round(_mm512_load_si512(idx_store + 64), islen_store64[1]);
			if(num_sym > 128) split_round(_mm512_load_si512(idx_store + 128), islen_store64[2]);
			if(num_sym > 192) split_round(_mm512_load_si512(idx_store + 192), islen_store64[3]);
			if(num_sym > 256) {
				auto idx = _mm256_load_si256(reinterpret_cast<const __m256i*>(idx_store + 256));
				const uint32_t I_LAST_MASK = ((1 << 30) -1);
				uint32_t islen = islen_store64[4] & I_LAST_MASK;
				auto elems = _mm512_castsi512_si256(shuffle256(_mm512_castsi256_si512(idx), sc0, sc1, sc2, sc3));
				elems = _mm256_mask_permutexvar_epi8(elems, _cvtu32_mask32(islen), idx, sc4);
				auto lo_elems = _mm256_mask_testn_epi8_mask(LAST_MASK, elems, _mm512_castsi512_si256(test));
				uint32_t i_lo_elems = _cvtmask32_u32(lo_elems);
				int loel_cnt = _mm_popcnt_u32(i_lo_elems);
				compress_store_256_8(idx_store + lo_pos, lo_elems, idx);
				compress_store_256_8(idx_store + hi_pos, _knot_mask32(lo_elems), idx);
				uint32_t lo_islen = _pext_u32(islen, i_lo_elems);
				uint32_t hi_islen = _pext_u32(islen, ~i_lo_elems & I_LAST_MASK);
				bit_write32flush(islen_store + (lo_pos>>3), lo_islen, loel_cnt, lo_byte, lo_pos&7);
				bit_write32flush(islen_store + (hi_pos>>3), hi_islen, 30 - loel_cnt, hi_byte, hi_pos&7);
				lo_pos += loel_cnt;
				hi_pos += 30 - loel_cnt;
			}
		};
		
		for(int i=1; i<8; i++) {
			test = _mm512_add_epi8(test, test);
			sort_round(sc0l, sc1l, sc2l, sc3l, sc4l);
		}
		
		// sort through high halves
		test = _mm512_set1_epi8(1);
		for(int i=0; i<7; i++) {
			sort_round(sc0h, sc1h, sc2h, sc3h, sc4h);
			test = _mm512_add_epi8(test, test);
		}
		// it's unlikely that we'll need to sort through all bits, so shortcut past the final bit if we can
		bool continue_sort;
		if(num_sym <= 64) {
			auto finish_sort = _mm512_test_epi8_mask(sc0h, test);
			continue_sort = _ktestz_mask64_u8(finish_sort, finish_sort) == 0;
		} else if(num_sym <= 128) {
			continue_sort = _kortestz_mask64_u8(
				_mm512_test_epi8_mask(sc0h, test),
				_mm512_test_epi8_mask(sc1h, test)
			) == 0;
		} else {
			auto finish_sort = _mm512_testn_epi8_mask(sc0h, test);
			if(num_sym > 64)  finish_sort = _mm512_mask_testn_epi8_mask(finish_sort, sc1h, test);
			if(num_sym > 128) finish_sort = _mm512_mask_testn_epi8_mask(finish_sort, sc2h, test);
			if(num_sym > 192) finish_sort = _mm512_mask_testn_epi8_mask(finish_sort, sc3h, test);
			if(num_sym > 256) finish_sort = _mm512_mask_testn_epi8_mask(finish_sort, ZEXT256_512(sc4h), test);
			finish_sort = _knot_mask64(finish_sort); // probably neater than doing a ktestc with -1
			continue_sort = _ktestz_mask64_u8(finish_sort, finish_sort) == 0;
		}
		if(HEDLEY_UNLIKELY(continue_sort)) {
			sort_round(sc0h, sc1h, sc2h, sc3h, sc4h);
		}
		
		// write sorted info
		merge_lohi_store();
		// first, do indices
		if(num_sym > 256) {
			auto* islen_store32 = reinterpret_cast<const uint32_t*>(islen_store);
			for(int j=0; j<num_sym; j+=32) {
				auto idx = _mm512_cvtepu8_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(idx_store + j)));
				idx = _mm512_mask_add_epi16(idx, _cvtu32_mask32(islen_store32[j/32]), idx, _mm512_set1_epi16(0x100));
				_mm512_store_si512(idx_sorted + j, idx);
			}
		} else {
			for(int j=0; j<num_sym; j+=64)
				_mm512_store_si512(idx_sorted + j, _mm512_load_si512(idx_store + j));
		}
		// then histogram
		auto* islen_store64 = reinterpret_cast<const uint64_t*>(islen_store);
		for(int j=0; j<(num_sym>256 ? 256 : num_sym); j+=64) {
			auto idx = _mm512_load_si512(idx_store + j);
			auto islen = islen_store64[j/64];
			auto hist_lo = shuffle_idx(idx, sc0l, sc1l, sc2l, sc3l, islen, sc4l);
			auto hist_hi = shuffle_idx(idx, sc0h, sc1h, sc2h, sc3h, islen, sc4h);
			__m512i hist0, hist1;
			unpack_bytes(hist_lo, hist_hi, hist0, hist1);
			_mm512_storeu_si512(hist_sorted + j, hist0);
			_mm512_storeu_si512(hist_sorted + j + 32, hist1);
		}
		if(num_sym > 256) {
			const auto INTERLEAVE_HIST = VEC512_8((_x>>1) | ((_x&1)<<5));
			auto idx = _mm512_castsi256_si512(_mm256_load_si256(reinterpret_cast<const __m256i*>(idx_store + 256)));
			uint64_t islen = reinterpret_cast<const uint32_t*>(islen_store)[8];
			auto hist_lo = shuffle_idx(idx, sc0l, sc1l, sc2l, sc3l, islen, sc4l);
			auto hist_hi = shuffle_idx(idx, sc0h, sc1h, sc2h, sc3h, islen, sc4h);
			auto hist0 = _mm512_inserti64x4(hist_lo, _mm512_castsi512_si256(hist_hi), 1);
			hist0 = _mm512_permutexvar_epi8(INTERLEAVE_HIST, hist0);
			_mm512_storeu_si512(hist_sorted + 256, hist0);
		}
	}
	template<int num_sym>
	static void FixHistIdxLitlen(const uint8_t* idx_sorted, uint16_t* sym_index) {
		HEDLEY_STATIC_ASSERT(num_sym == 64 || num_sym == 128 || num_sym == 192 || num_sym == 256 || num_sym == 288, "Invalid number of elements to re-arrange");
		
		// split lo/hi counts
		__m512i hist0l, hist0h, hist1l, hist1h, hist2l, hist2h, hist3l, hist3h;
		hist1l = hist1h = hist2l = hist2h = hist3l = hist3h = _mm512_undefined_epi32();
		pack_bytes(
			_mm512_loadu_si512(sym_index),
			_mm512_loadu_si512(sym_index + 32),
			hist0l, hist0h
		);
		if(num_sym > 64) {
			pack_bytes(
				_mm512_loadu_si512(sym_index + 64),
				_mm512_loadu_si512(sym_index + 96),
				hist1l, hist1h
			);
		}
		if(num_sym > 128) {
			pack_bytes(
				_mm512_loadu_si512(sym_index + 128),
				_mm512_loadu_si512(sym_index + 160),
				hist2l, hist2h
			);
		}
		if(num_sym > 192) {
			pack_bytes(
				_mm512_loadu_si512(sym_index + 192),
				_mm512_loadu_si512(sym_index + 224),
				hist3l, hist3h
			);
		}
		
		for(int i=0; i<num_sym; i+=64) {
			auto idx = _mm512_load_si512(idx_sorted + i);
			auto hist_lo = ShuffleIdx<num_sym>(idx, hist0l, hist1l, hist2l, hist3l);
			auto hist_hi = ShuffleIdx<num_sym>(idx, hist0h, hist1h, hist2h, hist3h);
			__m512i histOut0, histOut1;
			unpack_bytes(hist_lo, hist_hi, histOut0, histOut1);
			_mm512_storeu_si512(sym_index + i, histOut0);
			_mm512_storeu_si512(sym_index + i + 32, histOut1);
		}
	}
	
	static int SortHist(const uint16_t* sym_counts, uint16_t* sym_index, uint16_t* hist_sorted, bool is_sample = false) {
		int skipped_sym = 0;
		if(size < 32) {
			auto valid_mask = _cvtu32_mask32((1ULL<<size)-1);
			auto sc = _mm512_mask_loadu_epi16(_mm512_set1_epi16(-1), valid_mask, sym_counts);
			
			if(max_length == 7) {
				// this is the header table - shouldn't be possible to have more than 9 bits to search through
				assert(_mm512_mask_cmpgt_epu16_mask(valid_mask, sc, _mm512_set1_epi16(511)) == 0);
			}
			
			auto test = _mm512_set1_epi16(1);
			auto idx = splice_i16(VEC512_16(_x), _mm512_testn_epi16_mask(sc, test));
			const int SHORT_ROUNDS = (max_length == 7) ? 6 : 13;
			const int MAX_ROUNDS = (max_length == 7) ? 9 : 16;
			for(int i=1; i<SHORT_ROUNDS; i++) {
				test = _mm512_add_epi16(test, test);
				idx = splice_i16(idx, _mm512_testn_epi16_mask(_mm512_permutexvar_epi16(idx, sc), test));
			}
			
			// it's unlikely we'll need to scan all bits, so bail early in most cases
			auto continue_sort = _mm512_mask_test_epi16_mask(
				valid_mask,
				sc, _mm512_set1_epi16(int16_t(0xffff<<SHORT_ROUNDS))
			);
			if(HEDLEY_UNLIKELY(!_ktestz_mask32_u8(continue_sort, continue_sort))) {
				for(int i=SHORT_ROUNDS; i<MAX_ROUNDS; i++) {
					test = _mm512_add_epi16(test, test);
					idx = splice_i16(idx, _mm512_testn_epi16_mask(_mm512_permutexvar_epi16(idx, sc), test));
				}
			}
			
			_mm512_store_si512(sym_index, idx);
			_mm512_store_si512(hist_sorted, _mm512_permutexvar_epi16(idx, sc));
		} else {
			assert(size == 286);
			
			const auto LAST_MASK = _cvtu32_mask32((1 << 30) -1);
			// unless we've got totally random data, there's a good chance that many symbols are unused
			// stripping these out means fewer items to sort
			auto cmp = _mm512_set1_epi16(is_sample ? 1 : 0);
			auto idx = VEC512_16(_x);
			alignas(64) uint16_t sym_index_packed[288];
			alignas(64) uint16_t hist_packed[288];
			for(int i=0; i<288; i+=32) {
				auto data = _mm512_load_si512(sym_counts + i);
				auto matched = i==256 ? _mm512_mask_cmpeq_epi16_mask(LAST_MASK, data, cmp) : _mm512_cmpeq_epi16_mask(data, cmp);
				compress_store_512_16(sym_index + skipped_sym, matched, idx);
				_mm512_storeu_si512(hist_sorted + skipped_sym, cmp);
				auto not_matched = _knot_mask32(matched);
				compress_store_512_16(sym_index_packed + i - skipped_sym, not_matched, idx);
				compress_store_512_16(hist_packed + i - skipped_sym, not_matched, data);
				
				skipped_sym += _mm_popcnt_u32(_cvtmask32_u32(matched));
				idx = _mm512_add_epi16(idx, _mm512_set1_epi16(32));
				
				// if sampling, we assume there's no zeroes
				if(is_sample)
					assert(_cvtmask32_u32(i==256 ? _mm512_mask_testn_epi16_mask(LAST_MASK, data, data) : _mm512_testn_epi16_mask(data, data)) == 0);
			}
			
			// strategy 1: split into sections (quicksort, partial bitonic?) and sort each
			// - has the benefit of not needing a full permute to get test elements
			// strategy 2: sort vectors and bitonic merge
			// strategy 3: full binary radix sort (implemented below)
			
			if(HEDLEY_LIKELY(skipped_sym >= 30)) {
				// merge packed lists into sorted - this is mostly to allow some of the excluded symbols to be used in the sort (necessary to round up to 64)
				for(int j=0; j<286-skipped_sym; j+=32) {
					auto mask = _cvtu32_mask32(_bzhi_u32(-1, 286-skipped_sym - j));
					_mm512_mask_storeu_epi16(
						hist_sorted + skipped_sym + j,
						mask,
						_mm512_load_si512(hist_packed + j)
					);
					_mm512_mask_storeu_epi16(
						sym_index + skipped_sym + j,
						mask,
						_mm512_load_si512(sym_index_packed + j)
					);
				}
				alignas(64) uint8_t sorted_idx[256];
				if(skipped_sym >= 286-64) {
					SortHistLitlen<64>(hist_sorted + 286-64, sorted_idx, hist_sorted + 286-64);
					
					auto hist0 = _mm512_loadu_si512(sym_index + 286-64);
					auto hist1 = _mm512_loadu_si512(sym_index + 286-64 + 32);
					auto si = _mm512_cvtepu8_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(sorted_idx)));
					si = _mm512_permutex2var_epi16(hist0, si, hist1);
					_mm512_storeu_si512(sym_index + 286-64, si);
					
					si = _mm512_cvtepu8_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(sorted_idx + 32)));
					si = _mm512_permutex2var_epi16(hist0, si, hist1);
					_mm512_storeu_si512(sym_index + 286-64 + 32, si);
				}
				else if(skipped_sym >= 286-128) {
					SortHistLitlen<128>(hist_sorted + 286-128, sorted_idx, hist_sorted + 286-128);
					FixHistIdxLitlen<128>(sorted_idx, sym_index + 286-128);
				}
				/* else if(skipped_sym >= 286-192) { // probably don't need to be so fine grained
					SortHistLitlen<192>(hist_sorted + 286-192, sorted_idx, hist_sorted + 286-192);
					FixHistIdxLitlen<192>(sorted_idx, sym_index + 286-192);
				} */
				else {
					SortHistLitlen<256>(hist_sorted + 30, sorted_idx, hist_sorted + 30);
					FixHistIdxLitlen<256>(sorted_idx, sym_index + 30);
				}
				return is_sample ? 0 : skipped_sym; // skip counting zeroes below
			} else {
				// not enough skipped items - do a full sort
				SortHistLitlen<288>(sym_counts, sym_index, hist_sorted);
			}
		}
		
#ifndef NDEBUG
		{ // double-check sorting
			uint16_t check_index[size], check_hist[size];
			for(int i=0; i<size; i++)
				check_index[i] = i;
			std::sort(check_index, check_index + size, [&](uint16_t a, uint16_t b) {
				// order by histogram value, then by symbol index
				if(sym_counts[a] == sym_counts[b])
					return a < b;
				return sym_counts[a] < sym_counts[b];
			});
			for(int i=0; i<size; i++)
				check_hist[i] = sym_counts[check_index[i]];
			
			assert(memcmp(sym_index, check_index, sizeof(check_index)) == 0);
			assert(memcmp(hist_sorted, check_hist, sizeof(check_hist)) == 0);
		}
#endif
		
		// skip zeroes
		while(size - skipped_sym > 0) {
			int remaining = size - skipped_sym;
			auto mask = _cvtu32_mask32((remaining & ~31) ? -1 : _bzhi_u32(-1, remaining));
			auto data = _mm512_maskz_loadu_epi16(mask, hist_sorted + skipped_sym);
			auto nonzero = _mm512_test_epi16_mask(data, data);
			auto count = _tzcnt_u32(_cvtmask32_u32(nonzero));
			skipped_sym += count;
			if(HEDLEY_LIKELY(count < 32)) break;
		}
		return skipped_sym;
	}
	
	// adopted from https://create.stephan-brumme.com/length-limited-prefix-codes/  [moffat.c]
	static void Moffat(unsigned numCodes, uint16_t* hist, uint8_t* bit_lengths) {
		// phase 1
		uint32_t sum_hist[size]; // use 32b sums to avoid 16b overflow
		unsigned int leaf = 0;
		unsigned int root = 0;
		for (unsigned next = 0; next < numCodes - 1; next++) {
			// first child (assign to hist[next])
			if (leaf >= numCodes || (root < next && sum_hist[root] < hist[leaf])) {
				sum_hist[next] = sum_hist[root];
				hist[root] = next;
				root++;
			} else {
				sum_hist[next] = hist[leaf];
				leaf++;
			}

			// second child (add to hist[next])
			if (leaf >= numCodes || (root < next && sum_hist[root] < hist[leaf])) {
				sum_hist[next] += sum_hist[root];
				hist[root] = next;
				root++;
			} else {
				sum_hist[next] += hist[leaf];
				leaf++;
			}
		}
		
		// phase 2
		// TODO: investigate if it's possible to do two items at a time, to reduce dep chain
		hist[numCodes - 2] = 0;
		for (int j = numCodes - 3; j >= 0; j--)
			hist[j] = hist[hist[j]] + 1;
		
		// phase 3
		int avail = 1;
		int root2 = numCodes - 1;
		int next = numCodes;
		
		if(size < 32) {
			auto depth = _mm256_setzero_si256();
			auto bl = _mm512_cvtepi16_epi8(_mm512_mask_loadu_epi16(_mm512_set1_epi8(-1), _bzhi_u32(-1, root2), hist));
			auto new_bl = _mm256_setzero_si256();
			while(avail) {
				int used = _mm_popcnt_u32(_cvtmask32_u32(_mm256_cmpeq_epi8_mask(bl, depth)));
				
				int fill = avail - used;
				if(fill > 0) {
					next -= fill;
					auto mask = _bzhi_u32(-1, fill) << next;
					new_bl = _mm256_mask_mov_epi8(new_bl, _cvtu32_mask32(mask), depth);
				}
				
				avail = 2 * used;
				depth = _mm256_add_epi8(depth, _mm256_set1_epi8(1));
			}
			_mm256_store_si256(reinterpret_cast<__m256i*>(bit_lengths), new_bl);
		} else {
			auto depth16 = _mm512_setzero_si512();
			auto depth8 = _mm512_setzero_si512();
			while(avail) {
				int used = 0, matched;
				do {
					auto bl = _mm512_loadu_si512(hist + root2 - 32);
					matched = _lzcnt_u32(_cvtmask32_u32(_mm512_cmpneq_epi16_mask(bl, depth16)));
					used += matched;
					root2 -= matched;
				} while(HEDLEY_UNLIKELY(matched == 32));
				unsigned fill = avail - used;
				if(fill > 0) {
					while(HEDLEY_UNLIKELY(fill > sizeof(__m512i))) {
						_mm512_storeu_si512(bit_lengths + next - sizeof(__m512i), depth8);
						next -= sizeof(__m512i);
						fill -= sizeof(__m512i);
					}
					_mm512_mask_storeu_epi8(
						bit_lengths + next - sizeof(__m512i),
						_cvtu64_mask64(~0ULL << (sizeof(__m512i) - fill)),
						depth8
					);
					next -= fill;
				}
				
				avail = 2 * used;
				depth16 = _mm512_add_epi16(depth16, _mm512_set1_epi16(1));
				depth8 = _mm512_adds_epu8(depth8, _mm512_set1_epi8(1));
			}
		}
	}
	
	// shorten using MiniZ algorithm; this is based on https://create.stephan-brumme.com/length-limited-prefix-codes/#miniz
	static void MinizShorten(uint16_t* histNumBits, unsigned numCodes, const uint8_t* bit_lengths) {
		HEDLEY_STATIC_ASSERT(max_length == 15 || max_length == 7, "Unsupported max_length");
		__m256i kraft_total256;
		if(max_length == 15) {
			__m512i count0 = _mm512_setzero_si512();
			__m512i count1 = _mm512_setzero_si512();
			__m512i count_total;
			if(size > 32) {
				loop_u8x64(numCodes, bit_lengths, [&](__m512i v_bitlen, uint64_t, size_t&) {
					count0 = _mm512_add_epi64(count0, _mm512_popcnt_epi64(_mm512_inserti64x4(
						_mm512_castsi256_si512(_mm256_inserti128_si256(
							_mm256_castsi128_si256(hist_match_pair(v_bitlen, 1)),
							hist_match_pair(v_bitlen, 5), 1
						)),
						_mm256_inserti128_si256(
							_mm256_castsi128_si256(hist_match_pair(v_bitlen, 9)),
							hist_match_pair(v_bitlen, 13), 1
						), 1
					)));
					count1 = _mm512_add_epi64(count1, _mm512_popcnt_epi64(_mm512_inserti64x4(
						_mm512_castsi256_si512(_mm256_inserti128_si256(
							_mm256_castsi128_si256(hist_match_pair(v_bitlen, 3)),
							hist_match_pair(v_bitlen, 7), 1
						)),
						_mm256_inserti128_si256(
							_mm256_castsi128_si256(hist_match_pair(v_bitlen, 11)),
							_mm_cvtsi64_si128(_cvtmask64_u64(
								_mm512_cmpgt_epi8_mask(v_bitlen, _mm512_set1_epi8(14))
							)), 1
						), 1
					)));
				});
				// combine counts
				count_total = _mm512_castps_si512(_mm512_shuffle_ps(
					_mm512_castsi512_ps(count0), _mm512_castsi512_ps(count1),
					_MM_SHUFFLE(2,0,2,0)
				));
			} else {
				// this is the distance tree
				assert(numCodes < 32);
				auto mask = _cvtu32_mask32(_bzhi_u32(-1, numCodes));
				auto v_bitlen = _mm512_broadcast_i64x4(_mm256_maskz_loadu_epi8(mask, bit_lengths));
				
				count_total = _mm512_popcnt_epi32(_mm512_inserti64x4(
					_mm512_castsi256_si512(_mm256_inserti128_si256(
						_mm256_castsi128_si256(hist_match_pair_x2(v_bitlen, 1)),
						hist_match_pair_x2(v_bitlen, 5), 1
					)), _mm256_inserti128_si256(
						_mm256_castsi128_si256(hist_match_pair_x2(v_bitlen, 9)),
						_mm_insert_epi32(
							_mm_cvtsi64_si128(_cvtmask64_u64(_mm512_cmpeq_epi8_mask(v_bitlen, VEC512_8(_x/32+13)))),
							_cvtmask32_u32(_mm256_cmpgt_epi8_mask(_mm512_castsi512_si256(v_bitlen), _mm256_set1_epi8(14))),
							2
						), 1
					), 1
				));
			}
			// add in the missing 0th entry via rotation
			count_total = _mm512_alignr_epi32(count_total, count_total, 15);
			_mm256_store_si256(reinterpret_cast<__m256i*>(histNumBits), _mm512_cvtepi32_epi16(count_total));
			
			auto kraft_total512 = _mm512_sllv_epi32(count_total, _mm512_set_epi32(
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
			));
			kraft_total256 = _mm256_add_epi32(
				_mm512_castsi512_si256(kraft_total512),
				_mm512_extracti64x4_epi64(kraft_total512, 1)
			);
		} else {
			// this is the dynamic Huffman length tree
			assert(numCodes < 32);
			auto mask = _cvtu32_mask32(_bzhi_u32(-1, numCodes));
			auto v_bitlen = _mm512_broadcast_i64x4(_mm256_maskz_loadu_epi8(mask, bit_lengths));
			
			auto count_total = _mm256_popcnt_epi32(_mm256_inserti128_si256(
				_mm256_castsi128_si256(hist_match_pair_x2(v_bitlen, 1)),
				_mm_insert_epi32(
					_mm_cvtsi64_si128(_cvtmask64_u64(_mm512_cmpeq_epi8_mask(v_bitlen, VEC512_8(_x/32+5)))),
					_cvtmask32_u32(_mm256_cmpgt_epi8_mask(_mm512_castsi512_si256(v_bitlen), _mm256_set1_epi8(6))),
					2
				), 1
			));
			count_total = _mm256_alignr_epi32(count_total, count_total, 7);
			_mm_store_si128(reinterpret_cast<__m128i*>(histNumBits), _mm256_cvtepi32_epi16(count_total));
			kraft_total256 = _mm256_sllv_epi32(count_total, _mm256_set_epi32(
				0, 1, 2, 3, 4, 5, 6, 7
			));
		}
		auto kraft_total128 = _mm_add_epi32(
			_mm256_castsi256_si128(kraft_total256),
			_mm256_extracti128_si256(kraft_total256, 1)
		);
		kraft_total128 = _mm_add_epi32(kraft_total128, _mm_unpackhi_epi64(kraft_total128,kraft_total128));
		kraft_total128 = _mm_add_epi32(kraft_total128, _mm_srli_epi64(kraft_total128, 32));
		auto kraft_total = _mm_cvtsi128_si32(kraft_total128);
		
		// iterate until Kraft sum doesn't exceed 1 anymore
		auto nodes_to_move = kraft_total & ((1 << max_length) -1);
		histNumBits[max_length] -= nodes_to_move;
		
		// TODO: this code needs to be tested more
		for(unsigned i = max_length - 1; i > 0; i--)
			if(HEDLEY_LIKELY(histNumBits[i] > 0)) {
				auto from_max_length = max_length - i;
				auto move_to_max_num = std::min<uint16_t>((uint64_t(nodes_to_move) * TO_MAX_DIVIDE[from_max_length]) >> 31, histNumBits[i]);
				histNumBits[i] -= move_to_max_num;
				histNumBits[max_length] += move_to_max_num << from_max_length;
				nodes_to_move -= (move_to_max_num << from_max_length) - move_to_max_num;
				
				if(!nodes_to_move) break;
				if(histNumBits[i] > 0) {
					// reached end, move one node down and distribute it
					histNumBits[i]--;
					histNumBits[i + 1] += 2;
					nodes_to_move--;
					while(nodes_to_move) {
						i++;
						from_max_length = max_length - i;
						assert(from_max_length > 0);
						auto move_down = (uint64_t(nodes_to_move + (1<<from_max_length)-2) * TO_MAX_DIVIDE[from_max_length]) >> 31;
						histNumBits[i] -= move_down;
						histNumBits[i + 1] += move_down*2;
						nodes_to_move -= move_down;
					}
					break;
				}
			}
	}
	
public:
	alignas(64) uint8_t lengths[(size + 63) & ~63];
	
	void CalcCodeLengths(const uint16_t* sym_counts, bool is_sample = false) {
		alignas(64) uint16_t sym_index[(size + 31) & ~31]; // TODO: can make this 8-bit for smaller tables
		alignas(64) uint16_t hist_sorted[(size + 31) & ~31];
		
		const auto INDICES_I16 = VEC512_16(_x);
		
		auto skipped_sym = SortHist(sym_counts, sym_index, hist_sorted, is_sample);
		int numCodes_ = size - skipped_sym;
		
		if(HEDLEY_UNLIKELY(numCodes_ <= 2)) {
			if(numCodes_ < 1) return; // invalid case
			
			memset(lengths, 0, sizeof(lengths));
			lengths[sym_index[skipped_sym]] = 1;
			if(numCodes_ == 2)
				lengths[sym_index[skipped_sym-1]] = 1;
			return;
		}
		
		unsigned numCodes = numCodes_;
		alignas(32) uint8_t bit_lengths[(size + 31) & ~31]; // compact bit lengths to 8 bits, since we don't care about really long lengths anyway (they'll immediately get shortened)
		Moffat(numCodes, hist_sorted + skipped_sym, bit_lengths);
		
		if(bit_lengths[0] > max_length) {
			alignas(32) uint16_t histNumBits[max_length + 1];
			MinizShorten(histNumBits, numCodes, bit_lengths);
			
			// generate bit lengths
			if(0 && size < 32) { // this seems to be slower than the naive method
				// TODO: is the following worth it?  probably is for distance table, might be for header table
				__m256i bl;
				auto slow_construct = [&]() {
					int sympos = skipped_sym;
					bl = _mm256_maskz_set1_epi8(_cvtu32_mask32(_bzhi_u32(-1, skipped_sym)), -max_length);
					for(int i=max_length; i>0; i--) {
						sympos += histNumBits[i];
						bl = _mm256_mask_add_epi8(bl, _cvtu32_mask32(_bzhi_u32(-1, sympos)), bl, _mm256_set1_epi8(1));
					}
				};
				
				if(max_length < 8) {
					auto hnb = _mm_load_si128(reinterpret_cast<const __m128i*>(histNumBits));
					hnb = _mm_insert_epi16(hnb, skipped_sym, 0);
					auto use_slow = _mm_cmpgt_epi16_mask(hnb, _mm_set1_epi16(8));
					if(HEDLEY_LIKELY(MASK8_TEST(_ktestz)(use_slow, use_slow))) {
						const auto EXPAND_POS = VEC512_8(_x<8 ? 0 : 16-(_x/8)*2);
						auto cmp = _mm512_cmpgt_epi8_mask(
							_mm512_permutexvar_epi8(EXPAND_POS, _mm512_castsi128_si512(hnb)),
							_mm512_set1_epi64(0x0706050403020100)
						);
						bl = _mm512_castsi512_si256(_mm512_maskz_compress_epi8(cmp, _mm512_srli_epi16(EXPAND_POS, 1)));
					} else {
						slow_construct();
					}
				} else {
					HEDLEY_STATIC_ASSERT(max_length < 16, "Unsupported max_length");
					histNumBits[0] = skipped_sym;
					auto hnb = _mm256_load_si256(reinterpret_cast<const __m256i*>(histNumBits));
					// TODO: consider making cutoff point at 12, which allows for 3 processing rounds (+1 for zeroes) instead of 4?
					auto use_slow = _mm256_cmpgt_epi16_mask(hnb, _mm256_set1_epi16(16));
					if(HEDLEY_LIKELY(MASK16_TEST(_ktestz)(use_slow, use_slow))) {
						auto hnb2 = _mm512_permutexvar_epi8(_mm512_set_epi32(
							0x02020202, 0x0a0a0a0a, 0x12121212, 0x1a1a1a1a,
							0x04040404, 0x0c0c0c0c, 0x14141414, 0x1c1c1c1c,
							0x06060606, 0x0e0e0e0e, 0x16161616, 0x1e1e1e1e,
							0x08080808, 0x10101010, 0x18181818, 0
						), _mm512_castsi256_si512(hnb));
						
						const auto EXPAND_16 = VEC512_8(_x&0xf);
						auto cmp = _mm512_cmpgt_epi8_mask(
							_mm512_shuffle_epi32(hnb2, _MM_PERM_AAAA), EXPAND_16
						);
						bl = _mm512_castsi512_si256(_mm512_maskz_compress_epi8(cmp, _mm512_set_epi32(
							0x0d0d0d0d, 0x0d0d0d0d, 0x0d0d0d0d, 0x0d0d0d0d,
							0x0e0e0e0e, 0x0e0e0e0e, 0x0e0e0e0e, 0x0e0e0e0e,
							0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f, 0x0f0f0f0f,
							0, 0, 0, 0
						)));
						uint32_t xpos = ~0U << _mm_popcnt_u64(_cvtmask64_u64(cmp));
						
						auto compress_idx = _mm512_set_epi32(
							0x09090909, 0x09090909, 0x09090909, 0x09090909,
							0x0a0a0a0a, 0x0a0a0a0a, 0x0a0a0a0a, 0x0a0a0a0a,
							0x0b0b0b0b, 0x0b0b0b0b, 0x0b0b0b0b, 0x0b0b0b0b,
							0x0c0c0c0c, 0x0c0c0c0c, 0x0c0c0c0c, 0x0c0c0c0c
						);
						
						cmp = _mm512_cmpgt_epi8_mask(_mm512_shuffle_epi32(hnb2, _MM_PERM_BBBB), EXPAND_16);
						bl = _mm256_mask_expand_epi8(
							bl,
							_cvtu32_mask32(xpos),
							_mm512_castsi512_si256(_mm512_maskz_compress_epi8(cmp, compress_idx))
						);
						compress_idx = _mm512_add_epi8(compress_idx, _mm512_set1_epi8(-4));
						xpos <<= _mm_popcnt_u64(_cvtmask64_u64(cmp));
						
						// repeat above 2 times
						cmp = _mm512_cmpgt_epi8_mask(_mm512_shuffle_epi32(hnb2, _MM_PERM_CCCC), EXPAND_16);
						bl = _mm256_mask_expand_epi8(
							bl,
							_cvtu32_mask32(xpos),
							_mm512_castsi512_si256(_mm512_maskz_compress_epi8(cmp, compress_idx))
						);
						compress_idx = _mm512_add_epi8(compress_idx, _mm512_set1_epi8(-4));
						xpos <<= _mm_popcnt_u64(_cvtmask64_u64(cmp));
						
						cmp = _mm512_cmpgt_epi8_mask(_mm512_shuffle_epi32(hnb2, _MM_PERM_DDDD), EXPAND_16);
						bl = _mm256_mask_expand_epi8(
							bl,
							_cvtu32_mask32(xpos),
							_mm512_castsi512_si256(_mm512_maskz_compress_epi8(cmp, compress_idx))
						);
					} else {
						slow_construct();
					}
				}
				
				auto si = _mm512_mask_loadu_epi16(INDICES_I16, _cvtu32_mask32((1U << size) - 1), sym_index);
				si = _mm512_or_si512(_mm512_slli_epi16(si, 8), _mm512_cvtepu8_epi16(bl));
				
				si = sort_hi32(si);
				_mm256_store_si256(
					reinterpret_cast<__m256i*>(lengths), _mm512_cvtepi16_epi8(si)
				);
				
				
				// strategy 2: sort, then compare and set bits
				/*
				int sympos = skipped_sym;
				auto bl = _mm256_maskz_set1_epi8(_mm256_cmplt_epu8_mask(idx, _mm256_set1_epi8(skipped_sym)), -max_length);
				for(int i=max_length; i>0; i--) {
					sympos += histNumBits[i];
					auto inc = _mm256_cmplt_epu8_mask(idx, _mm256_set1_epi8(sympos));
					bl = _mm256_mask_add_epi8(bl, inc, bl, _mm256_set1_epi8(1));
				}*/
				
			} else {
				// TODO: perhaps try vectorizing this
				for(int i=0; i<skipped_sym; i++)
					lengths[sym_index[i]] = 0;
				int ptr = skipped_sym;
				for(int i=max_length; i>0; i--) {
					for(unsigned j=0; j<histNumBits[i]; j++) {
						lengths[sym_index[ptr++]] = i;
					}
				}
			}
			
		} else {
			// re-arrange based on sorted index
			if(0 && size < 32) { // this is slower than the naive method
				
				// index inversion idea from IJzerbaard [https://www.reddit.com/r/simd/comments/xe0qdg/computing_the_inverse_permutationshuffle/ioewfcm/]
				auto si = _mm512_mask_expandloadu_epi8(
					_mm512_slli_epi16(INDICES_I16, 8),
					_cvtu64_mask64(0x5555555555555555 << (skipped_sym*2)),
					bit_lengths
				);
				#pragma GCC diagnostic push
				#pragma GCC diagnostic ignored "-Warray-bounds"
				si = _mm512_mask_loadu_epi8(
					si, _cvtu64_mask64(0xaaaaaaaaaaaaaaaa & ((1ULL<<(size*2))-1)),
					reinterpret_cast<const uint8_t*>(sym_index) -1
				);
				#pragma GCC diagnostic pop
				/* previous idea:
				auto bl = _mm256_maskz_expandloadu_epi8(_cvtu32_mask32(-1 << skipped_sym), bit_lengths);
				auto si = _mm512_mask_loadu_epi16(INDICES_I16, _cvtu32_mask32((1U << size) - 1), sym_index);
				si = _mm512_or_si512(_mm512_slli_epi16(si, 8), _mm512_cvtepu8_epi16(bl));
				*/
				
				si = sort_hi32(si);
				_mm256_store_si256(
					reinterpret_cast<__m256i*>(lengths), _mm512_cvtepi16_epi8(si)
				);
			} else {
				// TODO: perhaps try vectorizing this
				for(int i=0; i<skipped_sym; i++)
					lengths[sym_index[i]] = 0;
				for(unsigned i=0; i<numCodes; i++)
					lengths[sym_index[i+skipped_sym]] = bit_lengths[i];
			}
		}
	}
	
	void CalcCodes(uint16_t* codes) const { // TODO: allow codes type to be template'd down to uint8_t + better optimisations for max_length==7
		__m256i length_hist;
		// TODO: are the following vectorized hist-count worth it?
		if(size >= 64) {
			const auto INDICES = VEC512_8(_x);
			// VERPMB table equivalent to _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(x, 1))
			// index 0 is assumed to be 0 (true in all cases we use this)
			const auto PERM_EXPAND_HIGH = VEC512_16(_x+32);
			auto hist = _mm512_setzero_si512();
			uint16_t hist_last = 0;
			for(int i=0; i<size; i+=sizeof(__m512i)) {
				auto sym_len = _mm512_load_si512(lengths + i);
				
				auto code0 = _mm512_setzero_si512(); // hope compiler optimizes this away so that the first add uses maskz
				auto code1 = code0;
				
				auto cmp = _mm512_set1_epi8(1);
				auto perm = _mm512_set1_epi16(4);
				auto cmp_round = [&](__m512i lenh) -> uint64_t {
					auto len_match = _mm512_cmpeq_epi8_mask(sym_len, cmp);
					auto idx = _mm512_maskz_expand_epi8(len_match, INDICES);
					// TODO: consider sticking to 8b but detect overflows
					code0 = _mm512_mask_add_epi16(
						code0,
						len_match,
						_mm512_cvtepu8_epi16(_mm512_castsi512_si256(idx)),
						lenh
					);
					code1 = _mm512_mask_add_epi16(
						code1,
						_kshiftri_mask64(len_match, 32),
						_mm512_permutexvar_epi8(PERM_EXPAND_HIGH, idx),
						lenh
					);
					cmp = _mm512_add_epi8(cmp, _mm512_set1_epi8(1));
					return _cvtmask64_u64(len_match);
				};
				auto cmp_2round = [&]() -> __m128i {
					// TODO: consider pre-expanding hist to use VPERMD instead of VPERMW, or maybe pre-shuffle it around so PSHUFB works
					auto cmp0 = cmp_round(_mm512_permutexvar_epi16(perm, hist));
					perm = _mm512_add_epi16(perm, _mm512_set1_epi16(2));
					auto cmp1 = cmp_round(_mm512_permutexvar_epi16(perm, hist));
					perm = _mm512_add_epi16(perm, _mm512_set1_epi16(2));
					return _mm_insert_epi64(_mm_cvtsi64_si128(cmp0), cmp1, 1);
				};
				
				auto match1_2 = cmp_2round();
				auto match3_4 = cmp_2round();
				auto match5_6 = cmp_2round();
				auto match7_8 = cmp_2round();
				auto match9_10 = cmp_2round();
				auto match11_12 = cmp_2round();
				auto match13_14 = cmp_2round();
				auto histA = _mm512_popcnt_epi64(_mm512_inserti64x4(
					_mm512_castsi256_si512(_mm256_inserti128_si256(_mm256_setzero_si256(), match3_4, 1)),
					_mm256_inserti128_si256(_mm256_castsi128_si256(match7_8), match11_12, 1),
					1
				));
				auto histB = _mm512_popcnt_epi64(_mm512_inserti64x4(
					_mm512_castsi256_si512(_mm256_inserti128_si256(_mm256_castsi128_si256(match1_2), match5_6, 1)),
					_mm256_inserti128_si256(_mm256_castsi128_si256(match9_10), match13_14, 1),
					1
				));
				hist = _mm512_add_epi32(hist,
					_mm512_castps_si512(_mm512_shuffle_ps(
						_mm512_castsi512_ps(histA), _mm512_castsi512_ps(histB), _MM_SHUFFLE(2,0,2,0)
					))
				);
				
				// final match
				hist_last += _mm_popcnt_u64(cmp_round(_mm512_set1_epi16(hist_last)));
				
				_mm512_store_si512(codes + i, code0);
				if(size - i > 32) // check eliminated if the loop is unrolled
					_mm512_store_si512(codes + i + 32, code1);
			}
			length_hist = _mm512_cvtepi32_epi16(hist);
		} else if(size < 32) {
			// same as above, but duplicate the data, to allow us to do two counts per iteration
			auto sym_len = _mm512_broadcast_i64x4(_mm256_load_si256(reinterpret_cast<const __m256i*>(lengths)));
			auto code0 = _mm256_setzero_si256();
			auto cmp = VEC512_8(_x/32+1);
			
			const auto INDICES = VEC256_8(_x);
			auto cmp_2round = [&]() -> uint64_t {
				auto len_match = _mm512_cmpeq_epi8_mask(sym_len, cmp);
				code0 = _mm256_mask_expand_epi8(code0, len_match, INDICES);
				code0 = _mm256_mask_expand_epi8(code0, _kshiftri_mask64(len_match, 32), INDICES);
				cmp = _mm512_add_epi8(cmp, _mm512_set1_epi8(2));
				return _cvtmask64_u64(len_match);
			};
			auto cmp_4round = [&]() -> __m128i {
				auto cmp0 = cmp_2round();
				auto cmp1 = cmp_2round();
				return _mm_insert_epi64(_mm_cvtsi64_si128(cmp0), cmp1, 1);
			};
			auto matches0 = cmp_2round();
			auto matches1 = cmp_4round();
			auto matches01 = _mm256_inserti128_si256(
				_mm256_castsi128_si256(_mm_insert_epi64(_mm_setzero_si128(), matches0, 1)),
				matches1, 1
			);
			if(max_length < 8) {
				auto hist = _mm256_popcnt_epi32(matches01);
				length_hist = ZEXT128_256(_mm256_cvtepi32_epi16(hist));
			} else {
				HEDLEY_STATIC_ASSERT(max_length < 16, "Unsupported max_length");
				auto matches2 = cmp_4round();
				auto matches3 = cmp_4round();
				auto matches = _mm512_inserti64x4(
					_mm512_castsi256_si512(matches01),
					_mm256_inserti128_si256(_mm256_castsi128_si256(matches2), matches3, 1), 1
				);
				auto hist = _mm512_popcnt_epi32(matches);
				length_hist = _mm512_cvtepi32_epi16(hist);
			}
			// insert last match
			auto match = _mm256_cmpeq_epi8_mask(_mm512_castsi512_si256(sym_len), _mm512_castsi512_si256(cmp));
			code0 = _mm256_mask_expand_epi8(code0, match, INDICES);
			
			_mm512_store_si512(codes, _mm512_cvtepu8_epi16(code0));
		} else {
			assert(0);
			/*
			// the following code never runs; just left here for good measure
			for(int i=0; i<size-1; i+=2) {
				codes[i] = length_hist[lengths[i]]++;
				codes[i+1] = length_hist[lengths[i+1]]++;
			}
			if(size & 1) {
				codes[size-1] = length_hist[lengths[size-1]]++;
			}
			length_hist[0] = 0;
			*/
		}
		
		auto code_base = _mm256_add_epi16(length_hist, length_hist);
		
		code_base = _mm256_add_epi16(code_base, _mm256_bslli_epi128(_mm256_add_epi16(code_base, code_base), 2));
		code_base = _mm256_add_epi16(code_base, _mm256_bslli_epi128(_mm256_slli_epi16(code_base, 2), 4));
		code_base = _mm256_add_epi16(code_base, _mm256_bslli_epi128(_mm256_slli_epi16(code_base, 4), 8));
		code_base = _mm256_add_epi16(code_base, _mm256_sllv_epi16(
			_mm256_permutexvar_epi16(_mm256_set_epi16(7,7,7,7,7,7,7,7,0,0,0,0,0,0,0,0), code_base),
			_mm256_set_epi16(8,7,6,5,4,3,2,1, 0,0,0,0,0,0,0,0)
		));
		
		for(int i=0; i<size; i+=32) {
			auto sym_len = _mm512_cvtepu8_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(lengths + i)));
			auto sym_codes = _mm512_load_si512(codes + i);
			
			// zero empty codes - probably unnecessary, but do as a precaution
			auto valid_sym = _mm512_test_epi16_mask(sym_len, sym_len);
			sym_codes = _mm512_maskz_add_epi16(
				valid_sym, sym_codes,
				_mm512_permutexvar_epi16(sym_len, _mm512_castsi256_si512(code_base))
			);
			// bit-reverse codes
			sym_codes = _mm512_gf2p8affine_epi64_epi8(sym_codes, _mm512_set1_epi64(0x8040201008040201ULL), 0);
			sym_codes = _mm512_shldv_epi16(sym_codes, sym_codes, _mm512_add_epi16(sym_len, _mm512_set1_epi16(-8)));
			_mm512_store_si512(codes + i, sym_codes);
		}
	}
	
	size_t WriteLengthTable(uint8_t *data, uint16_t *data_hist, unsigned& count) const {
		// strip zeroes off the end
		assert(size - count < 32); // in all cases in deflate, there's never more than 29 zeroes that can be stripped
		auto endmask = _cvtu32_mask32((1 << (size - count)) -1);
		auto enddata = _mm256_maskz_loadu_epi8(endmask, lengths + count);
		auto nonzero = _cvtmask32_u32(_mm256_test_epi8_mask(enddata, enddata));
		count += 32-_lzcnt_u32(nonzero);
		
		
		uint8_t *data_ptr = data;
		
		// TODO: naive method is faster than this for distance tree, but there's likely more optimisations that can be done in this case
		
		const auto INVALID_DATA = VEC512_8(64+_x); // shouldn't match any symbol
		auto last_vlen = _mm512_castsi512_si128(INVALID_DATA);
		// TODO: probably don't need this masking loop
		loop_u8x64(count, lengths, INVALID_DATA, [&](__m512i vlen, uint64_t loadmask, size_t& srcpos) {
			auto vlen_shift = _mm512_mask_permutexvar_epi8(
				_mm512_castsi128_si512(last_vlen), _cvtu64_mask64(~1ULL), VEC512_8(_x-1), vlen
			);
			auto symrun = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(vlen, vlen_shift));
			auto zeroes = _cvtmask64_u64(_mm512_testn_epi8_mask(vlen, vlen));
			
			// handle zero runs and symbol runs separately
			symrun &= ~zeroes;
			
			auto end_run = _lzcnt_u64(~symrun) | _lzcnt_u64(~zeroes);
			if(end_run == 64) {
				// full vector run - need to handle this differently to cater for cross-vector runs
				// at this point, all bytes in vlen are guaranteed to be the same
				
				assert(count >= srcpos + 64); // if we're here, we can't be on a partial vector
				
				srcpos += sizeof(__m512i);
				do {
					assert(count - srcpos < 256);
					
					loadmask = _bzhi_u64(-1LL, count - srcpos);
					auto vlen2 = _mm512_mask_loadu_epi8(INVALID_DATA, _cvtu64_mask64(loadmask), lengths + srcpos);
					auto runlen = _tzcnt_u64(_cvtmask64_u64(_mm512_cmpneq_epi8_mask(vlen, vlen2)));
					end_run += runlen;
					srcpos += runlen;
					if(runlen != 64) break;
				} while(srcpos < count);
				srcpos -= sizeof(__m512i);
				
				// write run to output
				if(zeroes) {
					const uint32_t MAX_ZERO_RUN = 18 | ((127 +128) << 8);
					while(end_run >= 138+3) { // max 2 iterations
						memcpy(data_ptr, &MAX_ZERO_RUN, 2);
						data_ptr += 2;
						end_run -= 138;
					}
					if(end_run >= 138) {
						memcpy(data_ptr, &MAX_ZERO_RUN, 4);
						data_ptr += 2 + end_run - 138;
					} else if(end_run > 10) {
						*data_ptr++ = 18;
						*data_ptr++ = end_run-11 +128;
					} else if(end_run > 0) {
						assert(end_run >= 3);
						*data_ptr++ = 17;
						*data_ptr++ = end_run-3 +128;
					}
				} else {
					auto quot = end_run / 6;
					auto rem = end_run % 6;
					if(rem == 1 || rem == 2) {
						// just duplicate the character
						uint16_t writeOut = _mm_cvtsi128_si32(_mm512_castsi512_si128(vlen));
						memcpy(data_ptr, &writeOut, 2);
						data_ptr += rem;
					}
					else if(rem) {
						*data_ptr++ = 16;
						*data_ptr++ = rem - 3 +128; // actually should be |128, but rem>=3, so + works as well
					}
					assert(quot < 64);
					const auto MAX_SYM_RUN = _mm512_set1_epi16(int16_t(16 | ((3 +128) << 8)));
					if(quot >= 32) {
						_mm512_storeu_si512(data_ptr, MAX_SYM_RUN);
						data_ptr += sizeof(__m512i);
						quot -= 32;
					}
					_mm512_mask_storeu_epi16(data_ptr, _cvtu32_mask32(_bzhi_u32(-1, quot)), MAX_SYM_RUN);
					data_ptr += quot*2;
				}
				
				last_vlen = _mm_insert_epi8(last_vlen, lengths[srcpos+63], 0);
				return;
			}
			// shorten vector by last run to be able to handle the full run
			loadmask = _bzhi_u64(loadmask, 64-end_run);
			srcpos -= end_run;
			last_vlen = _mm_insert_epi8(last_vlen, lengths[srcpos+63], 0);
			
			
			// eliminate runs < 3 long; TODO: see if there's a better way to do this
			auto runmask2 = _mm_insert_epi64(_mm_cvtsi64_si128(symrun), zeroes, 1);
			runmask2 = _mm_ternarylogic_epi64(
				runmask2,
				_mm_srli_epi64(runmask2, 1),
				_mm_srli_epi64(runmask2, 2),
				0x80 // A&B&C
			);
			runmask2 = _mm_ternarylogic_epi64(
				runmask2,
				_mm_add_epi64(runmask2, runmask2),
				_mm_slli_epi64(runmask2, 2),
				0xFE // A|B|C
			);
			uint64_t runmask = _mm_cvtsi128_si64(runmask2) | _mm_extract_epi64(runmask2, 1);
			
			// determine run lengths by finding indices of boundaries
			// to avoid cross-over on sym/zero runs, we'll shorten the end boundary by one place (i.e. a bit mask covering the first and last bit of the run)
			auto runbounds = _mm_ternarylogic_epi64(
				_mm_add_epi64(runmask2, runmask2),
				runmask2,
				_mm_srli_epi64(runmask2, 1),
				0x4c  //  ~(A & C) & B
			);
			
			uint64_t bounds = _mm_cvtsi128_si64(runbounds) | _mm_extract_epi64(runbounds, 1);
			auto indices = _mm512_castsi512_si256(_mm512_mask_compress_epi8(
				_mm512_set1_epi32(-1),
				_cvtu64_mask64(bounds),
				VEC512_8(_x)
			)); // guaranteed to be at most 4 bytes per run, so we'll never have more than half the vector filled
			auto run_lengths = _mm256_subs_epu8(
				_mm256_adds_epu8(_mm256_srli_epi16(indices, 8), _mm256_set1_epi16(1)),
				indices
			);
			
			// divide run lengths by 6 (rouneded up, but since we'll eliminate mod 1/2 later on, only +3 instead of +5)
			auto rl_div6 = _mm256_mulhi_epu16(_mm256_add_epi16(run_lengths, _mm256_set1_epi16(3)), _mm256_set1_epi16(int16_t(65536/6 + 1)));
			auto rl_mod6 = _mm512_castsi512_si256(_mm512_maskz_permutexvar_epi8(
				MASK_ALTERNATE,
				_mm512_castsi256_si512(run_lengths), VEC512_8(_x==0 ? 4 : (_x%6 ? _x%6 : 6)) // 0 needs to be treated as 64
			));
			
			// if modulo is 1 or 2, shorten that run by modulo
			auto bound_start = _pdep_u64(0x55555555, bounds);
			auto issym_compact = _cvtu32_mask16(~_pext_u64(zeroes, bound_start));
			auto rl_mod_lt3 = _mm256_mask_cmplt_epi16_mask(issym_compact, rl_mod6, _mm256_set1_epi16(3));
			run_lengths = _mm256_mask_sub_epi16(
				run_lengths, rl_mod_lt3, run_lengths, rl_mod6
			);
			// need to also fix up runmask
			auto bound_end = bounds ^ bound_start;
			runmask ^= _pdep_u64(_cvtmask16_u32(rl_mod_lt3), bound_end);
			runmask ^= _pdep_u64(_cvtmask16_u32(_mm256_mask_cmpeq_epi16_mask(
				issym_compact, rl_mod6, _mm256_set1_epi16(2)
			)), bound_end >> 1);
			rl_mod6 = _mm256_mask_blend_epi16(rl_mod_lt3, rl_mod6, _mm256_set1_epi16(6));
			
			// TODO: see if expansion should be done above instead
			auto run_reduced_mask = _mm512_shldv_epi32(
				_mm512_setzero_si512(), _mm512_set1_epi32(-1),
				_mm512_cvtepu16_epi32(_mm256_mask_add_epi16(_mm256_set1_epi16(2), issym_compact, rl_div6, rl_div6))
			);
			// shift into position and combine
			// TODO: high likelihood that upper vectors are all 0, so consider adding an if condition
			auto run_idx = _mm512_maskz_cvtepu8_epi16(__mmask32(MASK_ALTERNATE), indices);
			run_reduced_mask = _mm512_or_si512(
				_mm512_shldv_epi64(
					_mm512_and_si512(run_reduced_mask, _mm512_set1_epi64(0xffffffff)), _mm512_setzero_si512(), run_idx
				),
				_mm512_sllv_epi64(
					_mm512_srli_epi64(run_reduced_mask, 32),
					_mm512_srli_epi64(run_idx, 32)
				)
			);
			// combine into a single mask
			auto run_reduced_mask256 = _mm256_or_si256(
				_mm512_castsi512_si256(run_reduced_mask),
				_mm512_extracti64x4_epi64(run_reduced_mask, 1)
			);
			auto run_reduced_mask128 = _mm_or_si128(
				_mm256_castsi256_si128(run_reduced_mask256),
				_mm256_extracti128_si256(run_reduced_mask256, 1)
			);
			uint64_t run_reduced_mask64 = _mm_cvtsi128_si64(run_reduced_mask128) | _mm_extract_epi64(run_reduced_mask128, 1);
			
			// fill masked bytes with max symbol repeat
			vlen = _mm512_mask_expand_epi8(vlen, _cvtu64_mask64(run_reduced_mask64), _mm512_set1_epi16(int16_t(0x8310))); // [16,3] (i.e. repeat previous symbol 6x)
			
			
			// set up 16-18 sym
			auto repeat_sym = _mm256_mask_blend_epi16(
				issym_compact,
				_mm256_add_epi16(run_lengths, _mm256_mask_blend_epi16(
					_mm256_cmpgt_epu16_mask(run_lengths, _mm256_set1_epi16(10)),
					_mm256_set1_epi16(0x117d), // [17,len-3 +128]
					_mm256_set1_epi16(0x1275)  // [18,len-11 +128]
				)),
				_mm256_add_epi16(rl_mod6, _mm256_set1_epi16(0x107d)) // [16,mod-3 +128]
			);
			repeat_sym = _mm256_shldi_epi16(repeat_sym, repeat_sym, 8);
			
			auto repeat_loc = bound_start | (bound_start << 1);
			
			// put into place
			vlen = _mm512_mask_expand_epi8(vlen, _cvtu64_mask64(repeat_loc), _mm512_castsi256_si512(repeat_sym));
			
			
			// compress down to final result
			auto wanted_bytes = run_reduced_mask64 | ~runmask;
			wanted_bytes &= loadmask;
			compress_store_512_8(data_ptr, _cvtu64_mask64(wanted_bytes), vlen);
			data_ptr += _mm_popcnt_u64(wanted_bytes);
		});
		
		size_t table_len = data_ptr - data;
		
		// do histogram
		auto histmask = _cvtu32_mask32((1<<19)-1);
		if(size < 32) {
			// count two symbols at once
			auto mask = _cvtu32_mask32(_bzhi_u32(-1, table_len));
			auto sym = _mm512_broadcast_i64x4(_mm256_mask_loadu_epi8(_mm256_set1_epi8(-128), mask, data));
			
			auto count0 = _mm512_popcnt_epi32(_mm512_inserti32x4(
				_mm512_castsi256_si512(_mm256_inserti128_si256(
					_mm256_castsi128_si256(hist_match_pair_x2(sym, 0)),
					hist_match_pair_x2(sym, 8), 1
				)),
				hist_match_pair_x2(sym, 16), 2
			));
			auto count1 = _mm256_popcnt_epi32(_mm256_inserti128_si256(
				_mm256_castsi128_si256(hist_match_pair_x2(sym, 4)),
				hist_match_pair_x2(sym, 12), 1
			));
			
			_mm512_mask_storeu_epi16(
				data_hist, histmask, 
				_mm512_add_epi16(
					_mm512_maskz_loadu_epi16(histmask, data_hist),
					_mm512_packus_epi32(count0, _mm512_castsi256_si512(count1))
				)
			);
		} else {
			auto count0_7 = _mm512_setzero_si512();
			auto count8_15 = _mm512_setzero_si512();
			auto count16_17 = _mm_setzero_si128();
			int count18 = 0;
			loop_u8x64(table_len, data, _mm512_set1_epi8(-128), [&](__m512i sym, uint64_t, size_t) {
				auto hist_match_8 = [](__m512i data, int n) -> __m512i {
					return _mm512_inserti64x4(
						_mm512_castsi256_si512(_mm256_inserti128_si256(
							_mm256_castsi128_si256(hist_match_pair(data, n)),
							hist_match_pair(data, n+2), 1
						)),
						_mm256_inserti128_si256(
							_mm256_castsi128_si256(hist_match_pair(data, n+4)),
							hist_match_pair(data, n+6), 1
						), 1
					);
				};
				count0_7 = _mm512_add_epi32(count0_7, _mm512_popcnt_epi64(hist_match_8(sym, 0)));
				count8_15 = _mm512_add_epi32(count8_15, _mm512_popcnt_epi64(hist_match_8(sym, 8)));
				count16_17 = _mm_add_epi32(count16_17, _mm_popcnt_epi64(hist_match_pair(sym, 16)));
				count18 += _mm_popcnt_u64(_cvtmask64_u64(_mm512_cmpgt_epi8_mask(sym, _mm512_set1_epi8(17))));
			});
			
			_mm512_mask_storeu_epi16(
				data_hist, histmask, 
				_mm512_add_epi16(
					_mm512_maskz_loadu_epi16(histmask, data_hist),
					_mm512_permutex2var_epi16(
						_mm512_packus_epi32(count0_7, count8_15),
						_mm512_set_epi16(
							1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
							34, 36, 32,
							30, 28, 22, 20, 14, 12, 6, 4,
							26, 24, 18, 16, 10, 8, 2, 0
						),
						_mm512_castsi128_si512(_mm_insert_epi32(count16_17, count18, 1))
					)
				)
			);
		}
		
		return table_len;
	}
	
	HuffmanTree(const uint16_t* sym_count, bool is_sample = false) {
		// zero out padding in lengths array
		_mm512_store_si512(lengths + (size & ~63), _mm512_setzero_si512());
		CalcCodeLengths(sym_count, is_sample);
	}
};

#endif
