
#include "common.hh"
#include "bitwriter.hh"
#include "lz77data.hh"


static HEDLEY_ALWAYS_INLINE __m512i lz77_hash32(__m512i data) {
	const uint32_t HASH_PROD = 0x9E3779B1; // 0x9E3779B1 = 2^32 * GoldenRatio
	__m512i hash_data;
	
	if(MATCH_TABLE_ORDER > 19)  // TODO: is this actually sensible?
		hash_data = _mm512_mullo_epi32(data, _mm512_set1_epi32(HASH_PROD));
	else
		// avoid VPMULLD as it is considered "AVX-512 heavy" (and generally slower than VPMADDWD)
		hash_data = _mm512_madd_epi16(data, _mm512_set1_epi32(HASH_PROD));
	hash_data = _mm512_and_si512(hash_data, _mm512_set1_epi32((1<<MATCH_TABLE_ORDER)-1));
	// TODO: see if above AND can be replaced with a shift
	return hash_data;
}

#ifdef NDEBUG
#define ASSUME HEDLEY_ASSUME
#else
#define ASSUME assert
#endif


constexpr uint64_t bitdep_gen(unsigned bits_to_set, unsigned bits_available) {
	uint64_t bits = 0;
	for (unsigned i = 0; i < bits_to_set; i++) {
		bits |= 1ULL << ((i * bits_available *2 + bits_to_set) / (bits_to_set*2));
	}
	return bits;
}
constexpr uint64_t bitset_gen(unsigned n) {
	if(n <= 16) // table for 0-16 bits set
		return bitdep_gen(16-n, 64-n);
	else // table for 17-32 bits set
		return bitdep_gen(32-n, 64-n);
}
static constexpr auto BITSET_TABLE = lut<33>(bitset_gen);
constexpr uint64_t bitextr32_gen(unsigned n) {
	// table for 33-64 bits set
	return bitdep_gen(32, n+32);
}
static constexpr auto BITEXTR_TABLE = lut<32>(bitextr32_gen);



static HEDLEY_ALWAYS_INLINE void lz77_find_twomatch(
	__m512i source_data, __m512i indices, uint32_t* match_offsets, uint16_t search_base_offset, const void* search_base,
	__mmask16& match1, __mmask16& match2, __m512i& wrapped_offsets
) {
	auto hash_data = lz77_hash32(source_data);
	
	// gather offsets from hash table
	auto offsets = _mm512_i32gather_epi32(hash_data, match_offsets, sizeof(uint32_t));
	auto sb_offset_v = _mm512_set1_epi32(search_base_offset);
	auto current_offsets = _mm512_add_epi16(indices, sb_offset_v);
	// handle wrap-around
	auto wrap = _mm512_cmpge_epu16_mask(offsets, _mm512_shuffle_epi8(current_offsets, _mm512_set4_epi32(
		0x0d0c0d0c, 0x09080908, 0x05040504, 0x01000100
	)));
	wrapped_offsets = _mm512_mask_sub_epi16(offsets, wrap, offsets, _mm512_set1_epi16(1<<WINDOW_ORDER));
	
	// scatter current indices to hash table
	offsets = _mm512_shrdi_epi32( // since this doesn't check for conflicts, we could lose an offset
		offsets, current_offsets, 16
	);
	_mm512_i32scatter_epi32(match_offsets, hash_data, offsets, sizeof(uint32_t));
	
	// gather actual data
	auto offsets1 = _mm512_srai_epi32(_mm512_slli_epi32(wrapped_offsets, 16), 16);
	auto offsets2 = _mm512_srai_epi32(wrapped_offsets, 16);
	auto compare_data1 = _mm512_i32gather_epi32(offsets1, search_base, 1);
	auto compare_data2 = _mm512_i32gather_epi32(offsets2, search_base, 1);
	
	// check equality
	match1 = _mm512_cmpeq_epi32_mask(source_data, compare_data1);
	match2 = _mm512_cmpeq_epi32_mask(source_data, compare_data2);
	
#ifdef LZ77_USE_CONFLICT // disabled due to slow VPCONFLICTD instruction on Intel
	// TODO: can probably do this better
	if(!match1) {
		auto conflicts = _mm512_conflict_epi32(source_data);
		match1 = _mm512_test_epi32_mask(conflicts, conflicts);
		auto matchloc = _mm512_xor_si512(_mm512_lzcnt_epi32(conflicts), _mm512_set1_epi32(31));
		auto matchoffsets = _mm512_permutexvar_epi32(matchloc, current_offsets);
		wrapped_offsets = _mm512_mask_blend_epi16(__mmask32(MASK_ALTERNATE), wrapped_offsets, matchoffsets);
	}
#endif
}

// like above, but don't search for matches (i.e. on first vector, which has no matches, and we want to avoid looking up a bad offset)
static HEDLEY_ALWAYS_INLINE void lz77_set_matches(
	__m512i source_data, __m512i indices, uint32_t* match_offsets, uint16_t search_base_offset
) {
	auto hash_data = lz77_hash32(source_data);
	auto current_offsets = _mm512_add_epi16(indices, _mm512_set1_epi32(search_base_offset));
	current_offsets = _mm512_slli_epi32(current_offsets, 16);
	_mm512_i32scatter_epi32(match_offsets, hash_data, current_offsets, sizeof(uint32_t));
	
	// TODO: consider in-vector matches
}

static const int LZ77_LITERAL_BITCOST_LOG2 = 2; // assume literals consume 4 bits on average (same value as libdeflate)
static HEDLEY_ALWAYS_INLINE __m512i vec512_16_log2(__m512i v) {
	// reverse bits
	v = _mm512_shldi_epi16(v, v, 8);
	v = _mm512_gf2p8affine_epi64_epi8(
		v, _mm512_set1_epi64(0x8040201008040201ULL), 0
	);
	
	// set high bits
	auto neg_v = _mm512_sub_epi16(_mm512_setzero_si512(), v); // _mm512_sign_epi16 is missing
	v = _mm512_or_si512(v, neg_v);
	
	return _mm512_popcnt_epi16(v);
}

static HEDLEY_ALWAYS_INLINE __m256i lz77_get_match_short(int num_match, __m256i compressed_idx, __m512i data0, __m512i data1, __m512i compressed_offsets, const uint8_t* offset_base) {
	assert(num_match > 0 && num_match <= 32);
	
	auto perm_idx_idx = VEC512_8(_x/8);
	const auto SPREAD_IDX = VEC512_8((_x&7) +4); // +4 to go past already matched 4 bytes
	
	auto do_match8 = [&](__m256i offs) -> uint64_t {
		auto cmp_offs = _mm512_i32gather_epi64(offs, offset_base, 1);
		auto cmp_idx = _mm512_permutex2var_epi8(
			data0,
			// TODO: could replace VPERMB here with a PSHUFB, if pre-arranged
			_mm512_add_epi8(_mm512_permutexvar_epi8(perm_idx_idx, _mm512_castsi256_si512(compressed_idx)), SPREAD_IDX),
			data1
		);
		perm_idx_idx = _mm512_add_epi8(perm_idx_idx, _mm512_set1_epi8(8));
		return _cvtmask64_u64(_mm512_cmpeq_epi8_mask(cmp_idx, cmp_offs));
	};
	// TODO: investigate smaller gathers
	
	switch((num_match+7) >> 3) {
		case 1: {
			return _mm256_castsi128_si256(_mm_cvtsi64_si128(
				do_match8(_mm256_cvtepi16_epi32(_mm512_castsi512_si128(compressed_offsets)))
			));
		} case 2: {
			auto offs1 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(compressed_offsets));
			uint64_t cmp0 = do_match8(_mm512_castsi512_si256(offs1));
			uint64_t cmp1 = do_match8(_mm512_extracti64x4_epi64(offs1, 1));
			return _mm256_castsi128_si256(_mm_insert_epi64(
				_mm_cvtsi64_si128(cmp0), cmp1, 1
			));
		} case 3: {
			auto offs1 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(compressed_offsets));
			auto offs2 = _mm256_cvtepi16_epi32(_mm512_extracti32x4_epi32(compressed_offsets, 2));
			uint64_t cmp0 = do_match8(_mm512_castsi512_si256(offs1));
			uint64_t cmp1 = do_match8(_mm512_extracti64x4_epi64(offs1, 1));
			uint64_t cmp2 = do_match8(offs2);
			return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_insert_epi64(
				_mm_cvtsi64_si128(cmp0), cmp1, 1
			)), _mm_cvtsi64_si128(cmp2), 1);
		} case 4: {
			auto offs1 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(compressed_offsets));
			auto offs2 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(compressed_offsets, 1));
			uint64_t cmp0 = do_match8(_mm512_castsi512_si256(offs1));
			uint64_t cmp1 = do_match8(_mm512_extracti64x4_epi64(offs1, 1));
			uint64_t cmp2 = do_match8(_mm512_castsi512_si256(offs2));
			uint64_t cmp3 = do_match8(_mm512_extracti64x4_epi64(offs2, 1));
			return _mm256_inserti128_si256(
				_mm256_castsi128_si256(_mm_insert_epi64(_mm_cvtsi64_si128(cmp0), cmp1, 1)),
				_mm_insert_epi64(_mm_cvtsi64_si128(cmp2), cmp3, 1),
				1
			);
		} default: HEDLEY_UNREACHABLE();
	}
}
static __m128i lz77_get_match8_len_long(int num_match, __m128i compressed_idx, const uint8_t* idx_base, __m128i compressed_offsets, const uint8_t* offset_base) {
	assert(num_match > 0 && num_match <= 8);
	#define DO_MATCH(n) \
		_cvtmask64_u64(_mm512_cmpneq_epi8_mask( \
			_mm512_loadu_si512(idx_base + _mm_extract_epi16(compressed_idx, n)), \
			_mm512_loadu_si512(offset_base + int16_t(_mm_extract_epi16(compressed_offsets, n))) \
		))
	#define DO_MATCH_PAIR(n) \
		_mm_insert_epi64( \
			_mm_cvtsi64_si128(DO_MATCH(n)), \
			DO_MATCH(n+1), 1 \
		)
	uint64_t match1 = DO_MATCH(0);
	__m512i match_bits;
	switch(num_match) { // TODO: quantise this more?
		case 1:
			return _mm_cvtsi32_si128(_tzcnt_u64(match1));
		case 2:
			return _mm_insert_epi8(
				_mm_cvtsi32_si128(_tzcnt_u64(match1)), _tzcnt_u64(DO_MATCH(1)), 1
			);
		case 3:
			return _mm_insert_epi8(_mm_insert_epi8(
				_mm_cvtsi32_si128(_tzcnt_u64(match1)), _tzcnt_u64(DO_MATCH(1)), 1
			), _tzcnt_u64(DO_MATCH(2)), 2);
		case 4: {
			auto match01 = _mm_insert_epi8(
				_mm_cvtsi32_si128(_tzcnt_u64(match1)), _tzcnt_u64(DO_MATCH(1)), 1
			);
			uint16_t match23 = _tzcnt_u64(DO_MATCH(2)) | (_tzcnt_u64(DO_MATCH(3)) << 8);
			
			return _mm_insert_epi16(match01, match23, 1);
		} case 5:
			#define DO_MATCH03 \
				_mm512_castsi256_si512(_mm256_inserti128_si256( \
					_mm256_castsi128_si256(_mm_insert_epi64( \
						_mm_cvtsi64_si128(match1), DO_MATCH(1), 1 \
					)), \
					DO_MATCH_PAIR(2), 1 \
				))
			match_bits = _mm512_inserti32x4(
				DO_MATCH03, _mm_cvtsi64_si128(DO_MATCH(4)), 2
			);
			break;
		case 6:
			match_bits = _mm512_inserti32x4(
				DO_MATCH03, DO_MATCH_PAIR(4), 2
			);
			break;
		case 7:
			match_bits = _mm512_inserti64x4(DO_MATCH03, _mm256_inserti128_si256(
				_mm256_castsi128_si256(DO_MATCH_PAIR(4)),
				_mm_cvtsi64_si128(DO_MATCH(6)), 1
			), 1);
			break;
		case 8:
			match_bits = _mm512_inserti64x4(DO_MATCH03, _mm256_inserti128_si256(
				_mm256_castsi128_si256(DO_MATCH_PAIR(4)),
				DO_MATCH_PAIR(6), 1
			), 1);
			break;
		default: HEDLEY_UNREACHABLE();
	}
	#undef DO_MATCH03
	#undef DO_MATCH_PAIR
	#undef DO_MATCH
	
	auto match_lens = _mm512_popcnt_epi64(_mm512_andnot_si512(
		match_bits, _mm512_add_epi64(match_bits, _mm512_set1_epi64(-1))
	));
	return _mm512_cvtepi64_epi8(match_lens);
}

static __m256i lz77_get_match_len_long(size_t avail_len, __mmask32 long_matches, __m256i match_len, __m512i idx, const uint8_t* idx_base, __m512i offsets, const uint8_t* offset_base) {
	if(HEDLEY_UNLIKELY(avail_len < 12 + 2*sizeof(__m512i)))
		return match_len;
	
	uint32_t long_matches_i = _cvtmask32_u32(long_matches);
	
	auto compressed2_offsets = _mm512_maskz_compress_epi16(long_matches, offsets);
	auto compressed2_idx = _mm512_maskz_compress_epi16(long_matches, idx);
	
	__m256i match_len_long;
	int num_long_matches = _mm_popcnt_u32(long_matches_i);
	
	
	// TODO: investigate which of these two approaches is better (seems Clang prefers first, GCC prefers second)
	
	switch((num_long_matches+7) >> 3) {
		case 1:
			match_len_long = _mm256_castsi128_si256(lz77_get_match8_len_long(
				num_long_matches,
				_mm512_castsi512_si128(compressed2_idx), idx_base,
				_mm512_castsi512_si128(compressed2_offsets), offset_base
			));
			break;
		case 2:
			match_len_long = _mm256_castsi128_si256(_mm_unpacklo_epi64(
				lz77_get_match8_len_long(
					8,
					_mm512_castsi512_si128(compressed2_idx), idx_base,
					_mm512_castsi512_si128(compressed2_offsets), offset_base
				),
				lz77_get_match8_len_long(
					num_long_matches - 8,
					_mm256_extracti128_si256(_mm512_castsi512_si256(compressed2_idx), 1), idx_base,
					_mm256_extracti128_si256(_mm512_castsi512_si256(compressed2_offsets), 1), offset_base
				)
			));
			break;
		case 3:
			match_len_long = _mm256_castsi128_si256(_mm_unpacklo_epi64(
				lz77_get_match8_len_long(
					8,
					_mm512_castsi512_si128(compressed2_idx), idx_base,
					_mm512_castsi512_si128(compressed2_offsets), offset_base
				),
				lz77_get_match8_len_long(
					8,
					_mm256_extracti128_si256(_mm512_castsi512_si256(compressed2_idx), 1), idx_base,
					_mm256_extracti128_si256(_mm512_castsi512_si256(compressed2_offsets), 1), offset_base
				)
			));
			match_len_long = _mm256_inserti128_si256(match_len_long, lz77_get_match8_len_long(
				num_long_matches - 16,
				_mm512_extracti32x4_epi32(compressed2_idx, 2), idx_base,
				_mm512_extracti32x4_epi32(compressed2_offsets, 2), offset_base
			), 1);
			break;
		case 4: {
			match_len_long = _mm256_castsi128_si256(_mm_unpacklo_epi64(
				lz77_get_match8_len_long(
					8,
					_mm512_castsi512_si128(compressed2_idx), idx_base,
					_mm512_castsi512_si128(compressed2_offsets), offset_base
				),
				lz77_get_match8_len_long(
					8,
					_mm256_extracti128_si256(_mm512_castsi512_si256(compressed2_idx), 1), idx_base,
					_mm256_extracti128_si256(_mm512_castsi512_si256(compressed2_offsets), 1), offset_base
				)
			));
			auto tmp_cidx = _mm512_extracti64x4_epi64(compressed2_idx, 1);
			auto tmp_coff = _mm512_extracti64x4_epi64(compressed2_offsets, 1);
			match_len_long = _mm256_inserti128_si256(match_len_long, _mm_unpacklo_epi64(
				lz77_get_match8_len_long(
					8,
					_mm256_castsi256_si128(tmp_cidx), idx_base,
					_mm256_castsi256_si128(tmp_coff), offset_base
				),
				lz77_get_match8_len_long(
					num_long_matches - 24,
					_mm256_extracti128_si256(tmp_cidx, 1), idx_base,
					_mm256_extracti128_si256(tmp_coff, 1), offset_base
				)
			), 1);
			break;
		} default: HEDLEY_UNREACHABLE();
	}
	
	
	/*
	auto tmpc = _mm512_extracti64x4_epi64(compressed2_offsets, 1);
	auto coff0 = _mm512_castsi512_si128(compressed2_offsets);
	auto coff1 = _mm256_extracti128_si256(_mm512_castsi512_si256(compressed2_offsets), 1);
	auto coff2 = _mm256_castsi256_si128(tmpc);
	auto coff3 = _mm256_extracti128_si256(tmpc, 1);
	tmpc = _mm512_extracti64x4_epi64(compressed2_idx, 1);
	auto cidx0 = _mm512_castsi512_si128(compressed2_idx);
	auto cidx1 = _mm256_extracti128_si256(_mm512_castsi512_si256(compressed2_idx), 1);
	auto cidx2 = _mm256_castsi256_si128(tmpc);
	auto cidx3 = _mm256_extracti128_si256(tmpc, 1);
	
	__mmask64 cmp_match;
	__m128i match_len_tmp, match_len_tmp1 = _mm_undefined_si128(), match_len_tmp2 = _mm_undefined_si128();
	auto cmp_match_pair = _mm_undefined_si128();
	auto cmp_matches = _mm512_undefined_epi32();
	switch(num_long_matches) { // might make more sense to quantize this
		default: HEDLEY_UNREACHABLE();
		#define MATCH_CASE(n, coff, cidx) \
			HEDLEY_FALL_THROUGH; \
			case (n)+1: \
				cmp_match = _mm512_cmpeq_epi8_mask( \
					_mm512_loadu_si512(idx_base + _mm_extract_epi16(cidx, (n)&7)), \
					_mm512_loadu_si512(offset_base + int16_t(_mm_extract_epi16(coff, (n)&7))) \
				); \
				cmp_match_pair = _mm_insert_epi64(cmp_match_pair, _cvtmask64_u64(cmp_match), (n)&1)
		#define MATCH_CASE2(n, coff, cidx) \
			MATCH_CASE(n+1, coff, cidx); \
			MATCH_CASE(n, coff, cidx); \
			cmp_matches = _mm512_inserti32x4(cmp_matches, cmp_match_pair, ((n)/2)&3); \
			cmp_match_pair = _mm_setzero_si128()
		#define MATCH_CASE8(n, coff, cidx) \
			MATCH_CASE2(n+6, coff, cidx); \
			MATCH_CASE2(n+4, coff, cidx); \
			MATCH_CASE2(n+2, coff, cidx); \
			MATCH_CASE2(n, coff, cidx); \
			match_len_tmp = _mm512_cvtepi64_epi8(_mm512_popcnt_epi64(_mm512_andnot_si512( \
				_mm512_add_epi64(cmp_matches, _mm512_set1_epi64(1)), cmp_matches \
			)))
		
		MATCH_CASE8(24, coff3, cidx3);
		match_len_tmp2 = match_len_tmp;
		MATCH_CASE8(16, coff2, cidx2);
		match_len_tmp2 = _mm_unpacklo_epi64(match_len_tmp, match_len_tmp2);
		MATCH_CASE8(8, coff1, cidx1);
		match_len_tmp1 = match_len_tmp;
		MATCH_CASE8(0, coff0, cidx0);
		match_len_tmp1 = _mm_unpacklo_epi64(match_len_tmp, match_len_tmp1);
		match_len_long = _mm256_inserti128_si256(_mm256_castsi128_si256(match_len_tmp1), match_len_tmp2, 1);
		
		#undef MATCH_CASE8
		#undef MATCH_CASE2
		#undef MATCH_CASE
	}*/
	
	
	match_len_long = _mm256_maskz_expand_epi8(long_matches, match_len_long);
	match_len = _mm256_add_epi8(match_len, match_len_long);
	
	long_matches = _mm256_test_epi8_mask(match_len_long, _mm256_set1_epi8(64));
	long_matches_i = _cvtmask32_u32(long_matches);
	if(HEDLEY_UNLIKELY(long_matches_i != 0)) {
		// have at least one match that's >= 76B long - we'll only extend the first such instance and drop the rest
		// TODO: actually try to get the lowest idx item
		long_matches_i = _blsi_u32(long_matches_i);
		long_matches = _cvtu32_mask32(long_matches_i);
		int16_t long_offset = _mm_cvtsi128_si32(_mm512_castsi512_si128(_mm512_maskz_compress_epi16(long_matches, offsets)));
		unsigned long_idx = _mm_cvtsi128_si32(_mm512_castsi512_si128(_mm512_maskz_compress_epi16(long_matches, idx)));
		
		// the match we're dealing with now should be exactly 76B long
		assert(_mm_cvtsi128_si32(_mm256_castsi256_si128(_mm256_maskz_compress_epi8(long_matches, match_len))) == 8+64);
		
		int long_len = 8+64;
		for(int i=0; i<3; i++) {
			if(HEDLEY_UNLIKELY(avail_len < 12 + (i+3)*sizeof(__m512i)))
				break; // TODO: handle partial loads
			
			long_offset += sizeof(__m512i);
			long_idx += sizeof(__m512i);
			
			int len = _tzcnt_u64(_cvtmask64_u64(_mm512_cmpneq_epi8_mask(
				_mm512_loadu_si512(idx_base + long_idx),
				_mm512_loadu_si512(offset_base + long_offset)
			)));
			long_len += len;
			if(!(len & int(sizeof(__m512i)))) break;
		}
		if(long_len > 254) long_len = 254;  // DEFLATE's max match length is 258 (length is later saturated to this amount, so we could also threshold at 255 here)
		match_len = _mm256_mask_expand_epi8(match_len, long_matches, _mm256_castsi128_si256(_mm_cvtsi32_si128(long_len)));
		// TODO: also wipe the match mask after this, to lessen conflict resolution
	}
	return match_len;
}

static HEDLEY_ALWAYS_INLINE __m128i lz77_get_match_len(__m128i idx, __m512i data0, __m512i data1, __m512i offsets, const uint8_t* idx_base, const uint8_t* offset_base, __mmask16 match1, __mmask16 match2, size_t avail_len, __m256i& selected_offsets, __m256i& selected_value) {
	// TODO: for WINDOW_ORDER==15, offsets could incorrectly wrap when added to
	
	// TODO: consider detecting continuations?  not strictly necessary, as will be eliminated later, but reduces lookups
	
	auto match = _mm512_kunpackw(match2, match1);
	uint32_t match_i = _cvtmask32_u32(match);
	offsets = _mm512_permutexvar_epi16(VEC512_16(_x*2 + (_x>15)), offsets);
	
	auto idx_x2 = _mm256_inserti128_si256(_mm256_castsi128_si256(idx), idx, 1);
	auto compressed_idx = _mm256_maskz_compress_epi8(match, idx_x2);
	auto compressed_offsets = _mm512_maskz_compress_epi16(match, _mm512_add_epi16(offsets, _mm512_set1_epi16(4))); // skip past already matched 4 bytes
	
	int num_match = _mm_popcnt_u32(match_i);
	ASSUME(num_match > 0 && num_match <= 32);
	
	
	// extend 4B matches by 8 bytes
	auto match_bits = lz77_get_match_short(num_match, compressed_idx, data0, data1, compressed_offsets, offset_base);
	auto match_len_sub4 = _mm256_maskz_popcnt_epi8(
		_cvtu32_mask32(_bzhi_u32(-1, num_match)),
		_mm256_andnot_si256(
			_mm256_add_epi8(match_bits, _mm256_set1_epi8(1)), match_bits
		)
	);
	
	// if we have >=12 byte matches, extend them to their full length
	auto long_matches = _mm256_test_epi8_mask(match_len_sub4, _mm256_set1_epi8(8));
	uint32_t long_matches_i = _cvtmask32_u32(long_matches);
	// TODO: consider extending lengths only after conflict resolution?
	if(long_matches_i != 0) {
		match_len_sub4 = lz77_get_match_len_long(
			avail_len, long_matches, match_len_sub4,
			_mm512_add_epi16(_mm512_cvtepu8_epi16(compressed_idx), _mm512_set1_epi16(12)),
			idx_base,
			_mm512_add_epi16(compressed_offsets, _mm512_set1_epi16(8)),
			offset_base
		);
	}
	
	
	// determine actual offsets
	assert(idx_base - offset_base <= (1<<WINDOW_ORDER));
	auto idx_offsets = _mm512_add_epi16(_mm512_cvtepu8_epi16(idx_x2), _mm512_set1_epi16(idx_base-offset_base));
	// offsets need to be 0-based, so subtract an additional 1 from it
	// note that `idx_offsets - offsets - 1` == `idx_offsets + ~offsets`
	offsets = _mm512_add_epi16(
		idx_offsets,
		_mm512_ternarylogic_epi64(offsets, offsets, offsets, 0xf) // ~A
	);
	assert(_mm512_mask_cmpge_epu16_mask(match, offsets, _mm512_set1_epi16(1<<WINDOW_ORDER)) == 0);
	
	auto match_len_x = _mm256_maskz_expand_epi8(match, match_len_sub4);
	// resolve same idx matches
	// compute the match cost saving
	auto match_value = _mm512_shldi_epi16(_mm512_cvtepu8_epi16(match_len_x), _mm512_set1_epi16(-1), LZ77_LITERAL_BITCOST_LOG2);
	// use log2 of distance to get a rough cost of the extra bits
	match_value = _mm512_sub_epi16(match_value, vec512_16_log2(offsets));
	// TODO: see if there's a better way to do the following
	// (don't allow invalid items to be selected)
	match_value = _mm512_maskz_add_epi16(match, match_value, _mm512_set1_epi16(4 << LZ77_LITERAL_BITCOST_LOG2));
	
	auto match_value1 = _mm512_castsi512_si256(match_value);
	auto match_value2 = _mm512_extracti64x4_epi64(match_value, 1);
	
	auto swap = _mm256_cmpgt_epi16_mask(match_value2, match_value1);
	selected_offsets = _mm256_mask_mov_epi16(_mm512_castsi512_si256(offsets), swap, _mm512_extracti64x4_epi64(offsets, 1));
	selected_value = _mm256_mask_mov_epi16(match_value1, swap, match_value2);
	return _mm_mask_mov_epi8(_mm256_castsi256_si128(match_len_x), swap, _mm256_extracti128_si256(match_len_x, 1));
}

static __m256i lz77_get_match_len_x2(__m256i idx, __m512i data0, __m512i data1, __m512i offsets1, __m512i offsets2, const uint8_t* idx_base, const uint8_t* offset_base, __mmask32 match1, __mmask32 match2, size_t avail_len, __m512i& selected_offsets, __m512i& selected_value) {
	assert(idx_base - offset_base <= (1<<WINDOW_ORDER));
	
	auto match = _mm512_kunpackd(match2, match1);
	int num_match = _mm_popcnt_u64(_cvtmask64_u64(match));
	ASSUME(num_match > 0 && num_match <= 64);
	
	int num_match1 = _mm_popcnt_u32(_cvtmask32_u32(match1));
	
	auto tmp_offsets1 = _mm512_permutexvar_epi16(VEC512_16(_x*2 + (_x>15)), offsets1);
	auto tmp_offsets2 = _mm512_permutexvar_epi16(VEC512_16(_x*2 + (_x>15)), offsets2);
	offsets1 = _mm512_inserti64x4(tmp_offsets1, _mm512_castsi512_si256(tmp_offsets2), 1);
	offsets2 = _mm512_shuffle_i64x2(tmp_offsets1, tmp_offsets2, _MM_SHUFFLE(3,2,3,2));
	
	auto compressed_offsets1 = _mm512_add_epi16(offsets1, _mm512_set1_epi16(4));
	auto compressed_offsets2 = _mm512_add_epi16(offsets2, _mm512_set1_epi16(4));
	__m512i match_len_sub4;
	
	if(num_match <= 32) {
		// combine compressed_offsets into one
		auto compressed_offsets = _mm512_maskz_compress_epi16(match2, compressed_offsets2);
		auto shift_idx = _mm512_sub_epi16(VEC512_16(_x), _mm512_set1_epi16(num_match1));
		compressed_offsets = _mm512_permutexvar_epi16(shift_idx, compressed_offsets);
		compressed_offsets = _mm512_mask_compress_epi16(compressed_offsets, match1, compressed_offsets1);
		
		auto compressed_idx = _mm512_castsi512_si256(_mm512_maskz_compress_epi8(match, _mm512_inserti64x4(_mm512_castsi256_si512(idx), idx, 1)));
		
		auto match_bits = lz77_get_match_short(num_match, compressed_idx, data0, data1, compressed_offsets, offset_base);
		auto match_len32 = _mm256_maskz_popcnt_epi8(
			_cvtu32_mask32(_bzhi_u32(-1, num_match)),
			_mm256_andnot_si256(
				_mm256_add_epi8(match_bits, _mm256_set1_epi8(1)), match_bits
			)
		);
		
		auto long_matches = _mm256_test_epi8_mask(match_len32, _mm256_set1_epi8(8));
		uint32_t long_matches_i = _cvtmask32_u32(long_matches);
		if(long_matches_i != 0) {
			match_len32 = lz77_get_match_len_long(
				avail_len, long_matches, match_len32,
				_mm512_add_epi16(_mm512_cvtepu8_epi16(compressed_idx), _mm512_set1_epi16(12)),
				idx_base,
				_mm512_add_epi16(compressed_offsets, _mm512_set1_epi16(8)),
				offset_base
			);
		}
		match_len_sub4 = _mm512_castsi256_si512(match_len32);
	} else {
		assert(match1 != 0 && match2 != 0); // if > 32 matches, each 32-bit component is guaranteed to have at least one bit set
		
		int num_match2 = num_match - num_match1;
		auto compressed_idx1 = _mm256_maskz_compress_epi8(match1, idx);
		auto compressed_offs1 = _mm512_maskz_compress_epi16(match1, compressed_offsets1);
		auto compressed_idx2 = _mm256_maskz_compress_epi8(match2, idx);
		auto compressed_offs2 = _mm512_maskz_compress_epi16(match2, compressed_offsets2);
		auto match_bits1 = lz77_get_match_short(
			num_match1, compressed_idx1,
			data0, data1,
			compressed_offs1, offset_base
		);
		auto match_bits2 = lz77_get_match_short(
			num_match2, compressed_idx2,
			data0, data1,
			compressed_offs2, offset_base
		);
		
		auto match_bits = _mm512_inserti64x4(_mm512_castsi256_si512(match_bits1), match_bits2, 1);
		auto match_mask = _cvtu64_mask64(_bzhi_u32(-1, num_match1) | (uint64_t(_bzhi_u32(-1, num_match2)) << 32));
		match_len_sub4 = _mm512_maskz_popcnt_epi8(match_mask, _mm512_andnot_si512(
			_mm512_add_epi8(match_bits, _mm512_set1_epi8(1)), match_bits
		));
		
		auto long_matches = _mm512_test_epi8_mask(match_len_sub4, _mm512_set1_epi8(8));
		uint64_t long_matches_i = _cvtmask64_u64(long_matches);
		if(long_matches_i != 0) {
			__m512i match_len2 = _mm512_setzero_si512();
			// TODO: see if can merge into one call if num matches <= 32
			if(long_matches_i & 0xffffffff) {
				match_len2 = _mm512_castsi256_si512(lz77_get_match_len_long(
					avail_len, long_matches, _mm512_castsi512_si256(match_len_sub4),
					_mm512_add_epi16(_mm512_cvtepu8_epi16(compressed_idx1), _mm512_set1_epi16(12)),
					idx_base,
					_mm512_add_epi16(compressed_offs1, _mm512_set1_epi16(8)),
					offset_base
				));
			}
			long_matches = _kshiftri_mask64(long_matches, 32);
			if(!_ktestz_mask32_u8(long_matches, long_matches)) {
				match_len2 = _mm512_inserti64x4(match_len2, lz77_get_match_len_long(
					avail_len, long_matches, _mm512_extracti64x4_epi64(match_len_sub4, 1),
					_mm512_add_epi16(_mm512_cvtepu8_epi16(compressed_idx2), _mm512_set1_epi16(12)),
					idx_base,
					_mm512_add_epi16(compressed_offs2, _mm512_set1_epi16(8)),
					offset_base
				), 1);
			}
			match_len_sub4 = match_len2;
		}
		match_len_sub4 = _mm512_maskz_compress_epi8(match_mask, match_len_sub4);
	}
	
	
	auto idx_offsets = _mm512_add_epi16(_mm512_cvtepu8_epi16(idx), _mm512_set1_epi16(idx_base-offset_base));
	offsets1 = _mm512_add_epi16(
		idx_offsets,
		_mm512_ternarylogic_epi64(offsets1, offsets1, offsets1, 0xf) // ~A
	);
	offsets2 = _mm512_add_epi16(
		idx_offsets,
		_mm512_ternarylogic_epi64(offsets2, offsets2, offsets2, 0xf) // ~A
	);
	assert(_mm512_mask_cmpge_epu16_mask(match1, offsets1, _mm512_set1_epi16(1<<WINDOW_ORDER)) == 0);
	assert(_mm512_mask_cmpge_epu16_mask(match2, offsets2, _mm512_set1_epi16(1<<WINDOW_ORDER)) == 0);
	
	auto match_len_x = _mm512_maskz_expand_epi8(match, match_len_sub4);
	auto match_len_x1 = _mm512_castsi512_si256(match_len_x);
	auto match_len_x2 = _mm512_extracti64x4_epi64(match_len_x, 1);
	// resolve same idx matches
	// compute the match cost saving
	auto match_value1 = _mm512_shldi_epi16(_mm512_cvtepu8_epi16(match_len_x1), _mm512_set1_epi16(-1), LZ77_LITERAL_BITCOST_LOG2);
	auto match_value2 = _mm512_shldi_epi16(_mm512_cvtepu8_epi16(match_len_x2), _mm512_set1_epi16(-1), LZ77_LITERAL_BITCOST_LOG2);
	match_value1 = _mm512_sub_epi16(match_value1, vec512_16_log2(offsets1));
	match_value2 = _mm512_sub_epi16(match_value2, vec512_16_log2(offsets2));
	match_value1 = _mm512_maskz_add_epi16(match1, match_value1, _mm512_set1_epi16(4 << LZ77_LITERAL_BITCOST_LOG2));
	match_value2 = _mm512_maskz_add_epi16(match2, match_value2, _mm512_set1_epi16(4 << LZ77_LITERAL_BITCOST_LOG2));
	
	auto swap = _mm512_cmpgt_epi16_mask(match_value2, match_value1);
	selected_offsets = _mm512_mask_mov_epi16(offsets1, swap, offsets2);
	selected_value = _mm512_mask_mov_epi16(match_value1, swap, match_value2);
	return _mm256_mask_mov_epi8(match_len_x1, swap, match_len_x2);
}

template<typename T>
static HEDLEY_ALWAYS_INLINE void lz77_cmov(T& dest, bool cond, T new_val) {
	dest = HEDLEY_UNPREDICTABLE(cond) ? new_val : dest;
	/*auto cmask = T(cond) -1;
	dest ^= new_val;
	dest &= cmask;
	dest ^= new_val;
	//dest = (dest & cmask) | (new_val & ~cmask);
	*/
}

static uint32_t lz77_resolve_conflict_mask(__m256i& matched_lengths_sub3, __m256i match_indices, __m512i match_value, __mmask32 match, int& skip_til_index) {
	// generate comparison values
	auto match_start = _mm512_cvtepu8_epi16(match_indices);
	auto match_offset = _mm512_add_epi16(match_start, _mm512_set1_epi16(3));
	auto match_end = _mm512_add_epi16(_mm512_cvtepu8_epi16(matched_lengths_sub3), match_offset);
	// approximate comparison value for whether to shorten matches
	auto match_shorten = _mm512_srli_epi16(match_value, LZ77_LITERAL_BITCOST_LOG2);
	match_shorten = _mm512_add_epi16(match_shorten, match_start);
	
	// compress down so only relevant items are checked
	auto compressed_end = _mm512_maskz_compress_epi16(match, match_end);
	auto compressed_shorten = _mm512_maskz_compress_epi16(match, match_shorten);
	auto compressed_start = _mm512_maskz_compress_epi16(match, match_start);
	auto compressed_value = _mm512_maskz_compress_epi16(match, match_value);
	
	// transpose
	const auto PERM32 = _mm512_set_epi32(15, 11, 7, 3,  14, 10, 6, 2,  13, 9, 5, 1,  12, 8, 4, 0);
	compressed_end = _mm512_permutexvar_epi32(PERM32, compressed_end);
	compressed_shorten = _mm512_permutexvar_epi32(PERM32, compressed_shorten);
	compressed_start = _mm512_permutexvar_epi32(PERM32, compressed_start);
	compressed_value = _mm512_permutexvar_epi32(PERM32, compressed_value);
	
	auto tmp1 = _mm512_unpacklo_epi16(compressed_shorten, compressed_end);
	auto tmp2 = _mm512_unpackhi_epi16(compressed_shorten, compressed_end);
	auto tmp3 = _mm512_unpacklo_epi16(compressed_value, compressed_start);
	auto tmp4 = _mm512_unpackhi_epi16(compressed_value, compressed_start);
	
	// store out for main loop
	alignas(64) uint64_t mem_data[32 +4];
	_mm512_store_si512(mem_data, _mm512_unpacklo_epi32(tmp3, tmp1));
	_mm512_store_si512(mem_data + 8, _mm512_unpackhi_epi32(tmp3, tmp1));
	_mm512_store_si512(mem_data + 16, _mm512_unpacklo_epi32(tmp4, tmp2));
	_mm512_store_si512(mem_data + 24, _mm512_unpackhi_epi32(tmp4, tmp2));
	// extra padding
	_mm256_store_si256(reinterpret_cast<__m256i*>(mem_data + 32), _mm256_set1_epi16(-1));
	
	unsigned num_match = _mm_popcnt_u32(_cvtmask32_u32(match));
	assert(num_match > 0);
	
	assert(((mem_data[0] >> 16) & 0xff) >= skip_til_index);
	unsigned prev_i = 0;
	uint32_t selected = 0;
	for(unsigned i=1; i<num_match; i++) {
		ASSUME(prev_i < i);
		
		auto prev_data = _mm256_broadcastq_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(mem_data + prev_i)));
		auto cur_data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mem_data + i));
		prev_data = _mm256_shuffle_epi8(prev_data, _mm256_set_epi64x(  // rearrange to [value, start, end, end]
			0x0f0e'0f0e'0b0a'0908, 0x0706'0706'0302'0100,
			0x0f0e'0f0e'0b0a'0908, 0x0706'0706'0302'0100
		));
		cur_data = _mm256_shuffle_epi8(cur_data, _mm256_set_epi64x(  // rearrange to [value, start, shorten, start]
			0x0b0a'0d0c'0b0a'0908, 0x0302'0504'0302'0100,
			0x0b0a'0d0c'0b0a'0908, 0x0302'0504'0302'0100
		));
		auto diff = _mm256_sub_epi16(cur_data, prev_data);
		auto cmp = _mm256_cmpgt_epi16(diff, _mm256_set1_epi64x(
			0xffff'0003'0003'0001
			// start[i] >= end[prev_i]
			// ((value[i] + (1<<LZ77_LITERAL_BITCOST_LOG2)/2)>>LZ77_LITERAL_BITCOST_LOG2) - (end[prev_i] - start[i]) > 3   [3 = fudge factor for cost of len/dist symbols]
			// start[i] - start[prev_i] > 3
			// value[i] - value[prev_i] > 1
		));
		auto advance_i = _mm256_movemask_epi8(cmp) & 0x41414141;
		auto selcmp = _cvtmask16_u32(_mm256_cmpge_epu64_mask(cmp, _mm256_set1_epi64x(0x0000ffffffff0000)));
		
		int skip_i = _tzcnt_u32(advance_i | 0x80000000)>>3;
		i += skip_i;
		selected |= (_bzhi_u32(selcmp, skip_i+1) != 0) << prev_i;
		lz77_cmov(prev_i, advance_i, i);
		
		/*  if compiler refuses to use CMOV, this might be better
		if(advance_i) {
			// prev_i advances - find to where it does
			int skip_i = _tzcnt_u32(advance_i)>>3;
			i += skip_i;
			selected |= (_bzhi_u32(selcmp, skip_i+1) != 0) << prev_i;
			prev_i = i;
		} else {
			selected |= (selcmp!=0) << prev_i;
			i += 3;
		}
		*/
	}
	selected |= 1 << prev_i;
	skip_til_index = mem_data[prev_i] >> 48;
	
	uint32_t selected_match = _pdep_u32(selected, _cvtmask32_u32(match));
	auto selected_match_m = _cvtu32_mask32(selected_match);
	// check for overlong matches, due to length shortening
	auto cend = _mm512_maskz_compress_epi16(selected_match_m, match_end);
	auto cstart = _mm512_mask_compress_epi16(
		_mm512_set1_epi16(-1),  // ensure the last 'shorten' element is 0
		_cvtu32_mask32(_blsr_u32(selected_match)), // clearing lowest bit here will effectively shift it by 1 element
		match_start
	);
	
	auto shorten = _mm512_cvtepi16_epi8(_mm512_subs_epu16(cend, cstart));
	shorten = _mm256_maskz_expand_epi8(selected_match_m, shorten);
	assert(_mm256_cmplt_epu8_mask(matched_lengths_sub3, shorten) == 0);
	matched_lengths_sub3 = _mm256_sub_epi8(matched_lengths_sub3, shorten);
	
	return selected_match;
}

static HEDLEY_ALWAYS_INLINE __m512i lz77_symbol_encode(__m256i compressed_offsets, __m128i compressed_lengths, uint8_t* HEDLEY_RESTRICT& out_xbits_hi) {
	// length symbol (+128)
	auto len_sym = _mm256_castsi256_si128(_mm256_permutexvar_epi8(_mm256_castsi128_si256(compressed_lengths), _mm256_set_epi32(
		0x90909090, 0x8f8f8f8f, 0x8e8e8e8e, 0x8d8d8d8d,
		0x8c8c8b8b, 0x8a8a8989, 0x88878685, 0x84838281
	)));
	auto len_sym2 = _mm256_castsi256_si128(_mm256_permutexvar_epi8(_mm256_castsi128_si256(_mm_srli_epi16(compressed_lengths, 3)), _mm256_set_epi32(
		0x9c9c9c9c, 0x9b9b9b9b, 0x9a9a9a9a, 0x99999999,
		0x98989797, 0x96969595, 0x94939291, 0x8f8d8981
	)));
	len_sym = _mm_mask_max_epu8(
		_mm_set1_epi8(int8_t(0x9d)),
		_mm_cmpneq_epi8_mask(compressed_lengths, _mm_set1_epi8(-1)),
		len_sym, len_sym2
	);
	auto len_xbits_mask = _mm256_castsi256_si128(_mm256_permutexvar_epi8(_mm256_castsi128_si256(len_sym), _mm256_set_epi32(
		0x001f, 0x1f1f1f0f, 0x0f0f0f07, 0x07070703,
		0x03030301, 0x01010100, 0, 0
	)));
	auto len_with_xbits = _mm256_inserti128_si256(_mm256_castsi128_si256(len_sym), _mm_and_si128(
		len_xbits_mask, compressed_lengths
	), 1);
	
	
#if WINDOW_ORDER > 9
	auto dist_lzcnt = _mm512_cvtepi32_epi16(_mm512_lzcnt_epi32(_mm512_cvtepu16_epi32(compressed_offsets)));
	auto dist_log2 = _mm256_subs_epu16(_mm256_set1_epi16(31), dist_lzcnt);
	auto offset_sym_extra = _mm256_shrdv_epi16(
		compressed_offsets, compressed_offsets,
		_mm256_max_epu16(dist_log2, _mm256_set1_epi16(1)) // handle distance=0/1 case
	);
	auto dist_sym = _mm256_shldi_epi16(dist_log2, offset_sym_extra, 1+8); // bottom 8 bits are junk and will be discarded
	auto dist_xbits_mask = _mm256_shldv_epi16(_mm256_setzero_si256(), _mm256_set1_epi16(0x7fff), dist_log2);
	dist_sym = _mm256_or_si256(dist_sym, _mm256_set1_epi16(int16_t(0xa000)));
#else
	// alternative, lookup strat for smaller WINDOW_ORDER
	auto dist_sym = _mm512_castsi512_si256(_mm512_permutexvar_epi16(
		_mm512_castsi256_si512(compressed_offsets), _mm512_set_epi16(
			0xa900,0xa900,0xa900,0xa900,0xa900,0xa900,0xa900,0xa900,
			0xa800,0xa800,0xa800,0xa800,0xa800,0xa800,0xa800,0xa800,
			0xa700,0xa700,0xa700,0xa700,0xa600,0xa600,0xa600,0xa600,
			0xa500,0xa500,0xa400,0xa400,0xa300,0xa200,0xa100,0xa000
		)
	));
	auto dist_sym2 = _mm512_castsi512_si256(_mm512_permutexvar_epi16(
		_mm512_srli_epi16(_mm512_castsi256_si512(compressed_offsets), 4), _mm512_set_epi16(
			0xb100,0xb100,0xb100,0xb100,0xb100,0xb100,0xb100,0xb100,
			0xb000,0xb000,0xb000,0xb000,0xb000,0xb000,0xb000,0xb000,
			0xaf00,0xaf00,0xaf00,0xaf00,0xae00,0xae00,0xae00,0xae00,
			0xad00,0xad00,0xac00,0xac00,0xab00,0xaa00,0xa800,0xa000
		)
	));
	dist_sym = _mm256_max_epu16(dist_sym, dist_sym2);
	auto dist_xbits_mask = _mm256_shldv_epi16(
		_mm256_setzero_si256(),
		_mm256_set1_epi16(0x7fff),
		_mm256_srli_epi16(dist_sym, 1+8)
	);
	// instead of generating the mask, might be feasible to drag the xbits directly in?
#endif
	auto dist_xbits = _mm256_and_si256(dist_xbits_mask, compressed_offsets);
	/* // limit 7 bits in bottom xbits
	dist_xbits = _mm256_add_epi16(dist_xbits, dist_xbits);
	dist_xbits = _mm256_mask_avg_epu8(dist_xbits, __mmask16(MASK_ALTERNATE), dist_xbits, _mm256_setzero_si256());
	*/
	dist_xbits = _mm256_ternarylogic_epi64(
		dist_xbits, _mm256_slli_epi16(dist_xbits, 8), _mm256_set1_epi8(-128), 0xd8 // (B&C)|(A&~C)
	);
	auto dist_with_xbits = _mm256_shldi_epi16(dist_xbits, dist_sym, 8);
	auto dist_xbits_hi_mask = _mm256_cmpgt_epu8_mask(dist_sym, _mm256_set1_epi16(int16_t(0xb1ff)));
	compress_store_256_8(out_xbits_hi, dist_xbits_hi_mask, dist_xbits);
	out_xbits_hi += _mm_popcnt_u32(_cvtmask32_u32(dist_xbits_hi_mask));
	
	auto lendist = _mm512_inserti64x4(
		_mm512_castsi256_si512(len_with_xbits),
		dist_with_xbits, 1
	);
	return _mm512_permutexvar_epi8(_mm512_set_epi32(
		0x3f3e1f0f, 0x3d3c1e0e, 0x3b3a1d0d, 0x39381c0c,
		0x37361b0b, 0x35341a0a, 0x33321909, 0x31301808,
		0x2f2e1707, 0x2d2c1606, 0x2b2a1505, 0x29281404,
		0x27261303, 0x25241202, 0x23221101, 0x21201000
	), lendist);
}

static HEDLEY_ALWAYS_INLINE uint64_t lz77_literal_mask(__m128i compressed_lengths, __m128i compressed_idx, int num_match) {
	auto exp_clen = _mm512_cvtepu8_epi32(_mm_min_epu8(compressed_lengths, _mm_set1_epi8(63)));
	auto ld_mask1 = _mm512_maskz_shldv_epi64(
		MASK8(_cvtu32)(_bzhi_u32(-1, (num_match+1)>>1)),
		_mm512_set1_epi64(7), _mm512_set1_epi64(-1), exp_clen
	);
	auto ld_mask2 = _mm512_maskz_shldv_epi64(
		MASK8(_cvtu32)(_bzhi_u32(-1, num_match>>1)),
		_mm512_set1_epi64(7), _mm512_set1_epi64(-1), _mm512_srli_epi64(exp_clen, 32)
	);
	// shift to index pos
	auto exp_cidx = _mm512_cvtepu8_epi32(compressed_idx);
	ld_mask1 = _mm512_shldv_epi64(ld_mask1, _mm512_setzero_si512(), exp_cidx);
	ld_mask2 = _mm512_shldv_epi64(ld_mask2, _mm512_setzero_si512(), _mm512_srli_epi64(exp_cidx, 32));
	// or-reduce
	ld_mask1 = _mm512_or_si512(ld_mask1, ld_mask2);
	auto ld_mask256 = _mm256_or_si256(
		_mm512_castsi512_si256(ld_mask1),
		_mm512_extracti64x4_epi64(ld_mask1, 1)
	);
	// negate the mask in this or-reduction step
	auto ld_mask128 = _mm_ternarylogic_epi64(
		_mm256_castsi256_si128(ld_mask256),
		_mm256_extracti128_si256(ld_mask256, 1),
		_mm256_castsi256_si128(ld_mask256),
		3 // ~(A|B)
	);
	return _mm_cvtsi128_si64(ld_mask128) & _mm_extract_epi64(ld_mask128, 1);
}

template<bool first_vec>
static unsigned lz77_vec(__m512i data, __m512i data2, uint32_t* match_offsets, const uint8_t* HEDLEY_RESTRICT search_base, uint16_t search_base_offset, size_t avail_len, int& skip_til_index,
	uint8_t* HEDLEY_RESTRICT output, uint8_t* HEDLEY_RESTRICT& out_xbits_hi, BitWriter& is_lendist) {
	assert(skip_til_index >= 0 && skip_til_index < 258); // longest match length
	
	if(HEDLEY_UNLIKELY(skip_til_index >= int(sizeof(__m512i)))) {
		// completely skip these for now; maybe set matches later on?
		skip_til_index -= sizeof(__m512i);
		return 0;
	}
	
	uint64_t nonskip_mask = 0xffffffffffffffffULL << skip_til_index;
	
	// detect ranges 48-57 (0-9), 65-90 (A-Z), 97-122 (a-z)
	auto alphanum = _cvtmask64_u64(_mm512_test_epi8_mask(data, _mm512_permutexvar_epi8(
		_mm512_min_epu8(data, _mm512_set1_epi8(127)), // TODO: change to XOR 128
		_mm512_set_epi32(
			0, 0x00402020, 0x20202020, 0x20202020, 0x40404040, 0x40404040, 0x40404040, 0x40404000,
			0, 0x00404040, 0x40404040, 0x40404040, 0x40404040, 0x40404040, 0x40404040, 0x40404000
		)
	)));
	// also do whitespace changes
	auto whitespace = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(
		data, _mm512_shuffle_epi8( // TODO: can above permute be reused in some way?
			_mm512_set4_epi32(0x0000000d, 0x00000a09, 0x00000000, 0x00000020),
			data
		)
	));
	// TODO: consider adding boundaries based on upper/lower case
	
	// detect char/special boundaries
	uint64_t bounds = alphanum ^ (alphanum << 1);
	bounds |= whitespace ^ (whitespace << 1);
	unsigned num_bounds = _mm_popcnt_u64(bounds);
	// TODO: also consider going back a few spaces to try to find suffix matches like -"less ", etc??
	
	// TODO: consider run detection to deal with lack of intra-vector LZ77
	// maybe even 'conflict' detection?
	// or use the idea of gather (after scatter) to detect conflicts
	
	
	__mmask32 match;
	__m512i matched_offsets, match_value;
	__m256i matched_lengths_sub4;
	__m256i match_indices;
	
	auto skip_til_index_v32 = _mm512_set1_epi32(skip_til_index);
	
	// if boundary count <= 1/4
	// TODO: most text has 16-32 matches => maybe get rid of the <=16 case?
	if(num_bounds <= sizeof(__m512i) / 4) {
		// set some bits to reach the limit - this is because gather/scatter isn't any faster with masked out elements, so we may as well make full use of it
		// it also saves us from having to keep track of invalid elements
		bounds |= _pdep_u64(BITSET_TABLE[num_bounds], ~bounds);
		
		match_indices = _mm512_castsi512_si256(_mm512_maskz_compress_epi8(_cvtu64_mask64(bounds), VEC512_8(_x)));
		// expand indices 4x
		auto indices = _mm512_permutexvar_epi8(VEC512_8(_x/4), _mm512_castsi256_si512(match_indices));
		// shuffle data into place
		auto source_data = _mm512_permutex2var_epi8(data, _mm512_add_epi8(indices, _mm512_set1_epi32(0x03020100)), data2);
		indices = _mm512_and_si512(indices, _mm512_set1_epi32(0xff));
		
		__mmask16 match1, match2;
		__m512i offsets;
		if(first_vec) {
			lz77_set_matches(source_data, indices, match_offsets, search_base_offset);
		} else {
			lz77_find_twomatch(
				source_data, indices, match_offsets, search_base_offset, search_base,
				match1, match2, offsets
			);
			// filter out entries prior to skip_til_index
			match1 = _mm512_mask_cmpge_epi32_mask(match1, indices, skip_til_index_v32);
			match2 = _mm512_mask_cmpge_epi32_mask(match2, indices, skip_til_index_v32);
		}
		
		if(!first_vec && _kortestz_mask16_u8(match1, match2) == 0) {
			__m256i match_offs, match_val;
			matched_lengths_sub4 = _mm256_castsi128_si256(lz77_get_match_len(_mm256_castsi256_si128(match_indices), data, data2, offsets, search_base + search_base_offset, search_base, match1, match2, avail_len, match_offs, match_val));
			matched_offsets = _mm512_castsi256_si512(match_offs);
			match_value = _mm512_castsi256_si512(match_val);
			match = _kor_mask16(match1, match2);
		} else {
			// no match found, store and bail
			_mm512_mask_storeu_epi8(output - skip_til_index, _cvtu64_mask64(nonskip_mask), data);
			auto bytes_written = sizeof(__m512i)-skip_til_index;
			skip_til_index = 0;
			is_lendist.Skip(bytes_written);
			return bytes_written;
		}
	} else {
		if(num_bounds > sizeof(__m512i) / 2) {
			// too many items set, we'll just extract 32 of the set bits, evenly as possible
			// may arise from cases like `a=a+1;b=a*2;` or `1,1,1,1,1` etc
			bounds = _pdep_u64(BITEXTR_TABLE[num_bounds - sizeof(__m512i) / 2 -1], bounds);
		} else {
			bounds |= _pdep_u64(BITSET_TABLE[num_bounds], ~bounds);
		}
		
		// basically we do the 1/4 case above, but twice
		
		match_indices = _mm512_castsi512_si256(_mm512_maskz_compress_epi8(_cvtu64_mask64(bounds), VEC512_8(_x)));
		// expand indices 4x
		auto indices1 = _mm512_permutexvar_epi8(VEC512_8(_x/4), _mm512_castsi256_si512(match_indices));
		auto indices2 = _mm512_permutexvar_epi8(VEC512_8(_x/4+16), _mm512_castsi256_si512(match_indices));
		// shuffle data into place
		auto source_data1 = _mm512_permutexvar_epi8(_mm512_add_epi8(indices1, _mm512_set1_epi32(0x03020100)), data);
		auto source_data2 = _mm512_permutex2var_epi8(data, _mm512_add_epi8(indices2, _mm512_set1_epi32(0x03020100)), data2);
		indices1 = _mm512_and_si512(indices1, _mm512_set1_epi32(0xff));
		indices2 = _mm512_and_si512(indices2, _mm512_set1_epi32(0xff));
		
		__mmask16 match1, match2, match3, match4;
		__m512i offsets1, offsets2;
		if(first_vec) {
			lz77_set_matches(source_data1, indices1, match_offsets, search_base_offset);
		} else {
			lz77_find_twomatch(
				source_data1, indices1, match_offsets, search_base_offset, search_base,
				match1, match2, offsets1
			);
			match1 = _mm512_mask_cmpge_epi32_mask(match1, indices1, skip_til_index_v32);
			match2 = _mm512_mask_cmpge_epi32_mask(match2, indices1, skip_til_index_v32);
		}
		lz77_find_twomatch(
			source_data2, indices2, match_offsets, search_base_offset, search_base,
			match3, match4, offsets2
		);
		match3 = _mm512_mask_cmpge_epi32_mask(match3, indices2, skip_til_index_v32);
		match4 = _mm512_mask_cmpge_epi32_mask(match4, indices2, skip_til_index_v32);
		
		if(!first_vec && _kortestz_mask16_u8(match1, match2) == 0) {
			auto match31 = _mm512_kunpackw(match3, match1);
			auto match42 = _mm512_kunpackw(match4, match2);
			matched_lengths_sub4 = lz77_get_match_len_x2(match_indices, data, data2, offsets1, offsets2, search_base + search_base_offset, search_base, match31, match42, avail_len, matched_offsets, match_value);
			match = _kor_mask32(match31, match42);
		} else if(_kortestz_mask16_u8(match3, match4) == 0) {
			__m256i match_offs, match_val;
			match_indices = _mm256_castsi128_si256(_mm256_extracti128_si256(match_indices, 1));
			matched_lengths_sub4 = _mm256_castsi128_si256(lz77_get_match_len(_mm256_castsi256_si128(match_indices), data, data2, offsets2, search_base + search_base_offset, search_base, match3, match4, avail_len, match_offs, match_val));
			matched_offsets = _mm512_castsi256_si512(match_offs);
			match_value = _mm512_castsi256_si512(match_val);
			match = _kor_mask16(match3, match4);
			// because the bottom 16 are effectively skipped, we need to remove these
			bounds = _pdep_u64(0xffff0000, bounds);
		} else {
			// no match found, store and bail
			_mm512_mask_storeu_epi8(output - skip_til_index, _cvtu64_mask64(nonskip_mask), data);
			auto bytes_written = sizeof(__m512i)-skip_til_index;
			skip_til_index = 0;
			is_lendist.Skip(bytes_written);
			return bytes_written;
		}
	}
	
	// TODO: ideas: backwards searches? longer hash lookups?
	
	// matches cannot exceed 258 in length (whereas our strategy could exceed it by one byte)
	// we'll eventually need the length-3 value later on, so compute it by saturation
	auto matched_lengths_sub3 = _mm256_adds_epu8(matched_lengths_sub4, _mm256_set1_epi8(1));
	
	auto match_i = lz77_resolve_conflict_mask(matched_lengths_sub3, match_indices, match_value, match, skip_til_index);
	match = _cvtu32_mask32(match_i);
	int num_match = _mm_popcnt_u32(match_i);
	uint64_t idx_mask = _pdep_u64(match_i, bounds);
	
	assert(_mm_popcnt_u32(match_i) <= 16); // min 4B match len, hence there can be no more than 16 valid matches per 64 bytes
	
	// compress
	auto compressed_offsets = _mm512_castsi512_si256(_mm512_maskz_compress_epi16(match, matched_offsets));
	auto compressed_lengths = _mm256_castsi256_si128(_mm256_maskz_compress_epi8(match, matched_lengths_sub3));
	
	auto lendist = lz77_symbol_encode(compressed_offsets, compressed_lengths, out_xbits_hi);
	auto used_lendist = _mm512_cmpge_epu8_mask(
		_mm512_slli_epi16(lendist, 8), _mm512_set1_epi32(0xa4008900)
	);
	uint64_t used_lendist_i = _cvtmask64_u64(used_lendist);
	
	// expand into data vec
	assert(_mm_popcnt_u64(idx_mask & 0x1fffffffffffffff)*4 == _mm_popcnt_u64((idx_mask & 0x1fffffffffffffff)*15)); // bits should be at least 4 apart
	uint64_t idx_mask_expanded = idx_mask*15;
	data = _mm512_mask_expand_epi8(data, _cvtu64_mask64(idx_mask_expanded), lendist);
	
	assert((~nonskip_mask & idx_mask_expanded) == 0);
	
	uint64_t idx_lendist = _pdep_u64(_cvtmask64_u64(used_lendist_i), idx_mask_expanded);
	
	// construct eliminate masks
	uint64_t final_mask = lz77_literal_mask(
		compressed_lengths,
		_mm256_castsi256_si128(_mm256_maskz_compress_epi8(match, match_indices)),
		num_match
	);
	
	assert((final_mask & idx_mask_expanded) == 0);
	final_mask &= nonskip_mask;
	final_mask |= idx_lendist;
	
	// compress down unused bytes
	int final_bytes = _mm_popcnt_u64(final_mask);
	compress_store_512_8(output, _cvtu64_mask64(final_mask), data);
	is_lendist.Write64(_pext_u64(idx_lendist, final_mask), final_bytes);
	
	if(idx_mask >> 61) {
		// TODO: consider merging this into next lz77_vec's vector?
		// handle possible overflow
		used_lendist_i = _bzhi_u64(used_lendist_i, num_match*4); // remove bad items
		uint64_t overflow_elems = ~_pext_u64(idx_mask_expanded, idx_mask_expanded) & used_lendist_i;
		compress_store_512_8(output + final_bytes, _cvtu64_mask64(overflow_elems), lendist);
		int extra_bytes = _mm_popcnt_u64(overflow_elems);
		assert(extra_bytes <= 3);
		final_bytes += extra_bytes;
		is_lendist.Write57(_bzhi_u32(-1, extra_bytes), extra_bytes);
	}
	
	// update skip value
	skip_til_index -= sizeof(__m512i);
	if(skip_til_index < 0) skip_til_index = 0;
	
	return final_bytes;
}

template<class ChecksumClass>
static void lz77_encode(
	Lz77Data& output,
	// inputs (+gets changed)
	const void* HEDLEY_RESTRICT& src, size_t& len,
	uint16_t& window_offset, int& skip_next_bytes,
	uint32_t* match_offsets, bool is_first_block,
	// checksum callbacks
	ChecksumClass& cksum
) {
	BitWriter is_lendist_writer(output.is_lendist);
	uint8_t* xbits_hi_ptr = nullptr;
#if WINDOW_ORDER > 9
	xbits_hi_ptr = output.xbits_hi;
#endif
	auto _src = static_cast<const uint8_t*>(src);
	auto src_end = _src + len;
	
	assert((window_offset & (sizeof(__m512i)-1)) == 0);
	assert(output.len == 0);
	
	if(len > sizeof(__m512i)*2) {
		_src -= window_offset;
		len += window_offset;
		if(is_first_block) {
			assert(window_offset == 0);
			assert(skip_next_bytes == 0);
			// for the first buffer, there's an issue with the first 4 bytes matching itself (since all hash table entries point to it)
			// we avoid this issue by avoiding trying to find matches for the first vector (which would be pointless to do anyway)
			auto data = _mm512_loadu_si512(_src);
			auto next_data = _mm512_loadu_si512(_src + sizeof(__m512i));
			
			output.len += lz77_vec<true>(data, next_data, match_offsets, _src, 0, len-window_offset, skip_next_bytes,
				output.data + output.len, xbits_hi_ptr, is_lendist_writer);
			cksum.update(data);
			window_offset += sizeof(__m512i);
		}
		
		auto safe_end = reinterpret_cast<const uint8_t*>(uintptr_t(src_end - sizeof(__m512i)) & -int(sizeof(__m512i)));
		assert(_src < safe_end);
		while(_src < safe_end) {
			auto win_size = std::min(size_t(safe_end - _src), size_t(1<<WINDOW_ORDER));
			auto data = _mm512_loadu_si512(_src + window_offset);
			for(; window_offset < win_size && output.len < OUTPUT_BUFFER_SIZE - sizeof(__m512i)*2; window_offset += sizeof(__m512i)) {
				auto next_data = _mm512_loadu_si512(_src + window_offset + sizeof(__m512i));
				
				output.len += lz77_vec<false>(data, next_data, match_offsets, _src, window_offset, len-window_offset, skip_next_bytes,
					output.data + output.len, xbits_hi_ptr, is_lendist_writer);
				cksum.update(data);
				data = next_data;
			}
			_src += window_offset;
			len -= window_offset;
			window_offset &= (1<<WINDOW_ORDER) -1;
			
			if(output.len >= OUTPUT_BUFFER_SIZE - sizeof(__m512i)*2) break;
		}
	} else assert(len > 0);
	HEDLEY_STATIC_ASSERT(OUTPUT_BUFFER_SIZE > sizeof(__m512i)*2, "Output buffer must be large enough to hold at least two vectors");
	unsigned end_misalign = src_end - _src;
	if(end_misalign && output.len < OUTPUT_BUFFER_SIZE - sizeof(__m512i)*2) {
		// for now, just write the last part as literals; TODO: do LZ77 on the last part
		if(end_misalign >= sizeof(__m512i)) {
			auto data = _mm512_loadu_si512(_src);
			cksum.update(data);
			auto bytes_written = sizeof(__m512i)-skip_next_bytes;
			_mm512_mask_storeu_epi8(output.data + output.len - skip_next_bytes, _cvtu64_mask64(0xffffffffffffffffULL << skip_next_bytes), data);
			is_lendist_writer.Skip(bytes_written);
			end_misalign -= sizeof(__m512i);
			_src += sizeof(__m512i);
			//window_offset += sizeof(__m512i); // reached the end, window doesn't matter any more
			output.len += bytes_written;
			skip_next_bytes = 0;
		}
		if(end_misalign) {
			auto datamask = _bzhi_u64(-1LL, end_misalign);
			auto data = _mm512_maskz_loadu_epi8(_cvtu64_mask64(datamask), _src);
			cksum.update_partial(data, end_misalign);
			_mm512_mask_storeu_epi8(output.data + output.len - skip_next_bytes, _cvtu64_mask64(datamask ^ _bzhi_u64(-1LL, skip_next_bytes)), data);
			auto bytes_written = end_misalign-skip_next_bytes;
			is_lendist_writer.Skip(bytes_written);
			output.len += bytes_written;
			_src += end_misalign;
			//window_offset += end_misalign;
			skip_next_bytes = 0;
		}
		len = 0;
		window_offset &= (1<<WINDOW_ORDER) -1; // ...or just 0?
	}
	
	assert(output.len == is_lendist_writer.BitLength(output.is_lendist));
	
	// write end-of-block and pad buffer using 0s (saves having to deal with masking during later stages)
	_mm512_storeu_si512(output.data + output.len, ZEXT128_512(_mm_cvtsi32_si128(128)));
#if WINDOW_ORDER > 9
	_mm512_storeu_si512(xbits_hi_ptr, _mm512_setzero_si512());
	output.xbits_hi_len = xbits_hi_ptr - output.xbits_hi;
#endif
	is_lendist_writer.Write57(0xffffffff, 32);
	is_lendist_writer.Write57(0xffffffff, 32);
	
	
	src = _src;
	output.len++;
}
