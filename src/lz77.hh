
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

static HEDLEY_ALWAYS_INLINE __m128i lz77_get_match_len(__m512i idx, __m512i data0, __m512i data1, __m512i offsets, const uint8_t* idx_base, const uint8_t* offset_base, __mmask16 match1, __mmask16 match2, size_t avail_len, __m256i& selected_offsets, __m256i& selected_value) {
	alignas(64) int16_t off_mem[32];
	// TODO: for WINDOW_ORDER==15, off_mem could incorrectly wrap
	
	// TODO: consider detecting continuations?  not strictly necessary, as will be eliminated later, but reduces lookups
	
	auto match = _mm512_kunpackw(match2, match1);
	idx = _mm512_permutexvar_epi16(VEC512_16(_x*2), idx); // pack 32b -> 16b, and duplicate to upper 256b
	offsets = _mm512_permutexvar_epi16(VEC512_16(_x*2 + (_x>15)), offsets);
	
	auto compressed_idx = _mm512_maskz_compress_epi16(match, idx);
	auto compressed_offsets = _mm512_maskz_compress_epi16(match, _mm512_add_epi16(offsets, _mm512_set1_epi16(4))); // skip past already matched 4 bytes
	_mm512_store_si512(off_mem, compressed_offsets);
	
	auto perm_idx_idx = VEC512_8((_x/16) * 2);
	const auto SPREAD_IDX = VEC512_8((_x&15) +4); // +4 to go past already matched 4 bytes
	
	uint32_t match_i = _cvtmask32_u32(match);
	auto cmp0 = _mm512_undefined_epi32();
	auto cmp_matches = _mm512_undefined_epi32();
	auto cmp_match_pair = _mm_undefined_si128();
	
	int num_match = _mm_popcnt_u32(match_i);
	ASSUME(num_match > 0 && num_match <= 32);
	
	perm_idx_idx = _mm512_add_epi8(perm_idx_idx, _mm512_set1_epi8(((num_match-1) & ~3)*2));
	__m512i cmp_idx;
	
	switch(num_match) { // might make more sense to quantize this
		default: HEDLEY_UNREACHABLE();
		#define MATCH_CASE(n) \
			HEDLEY_FALL_THROUGH; \
			case (n)+1: \
				cmp0 = _mm512_inserti32x4(cmp0, _mm_loadu_si128(reinterpret_cast<const __m128i*>(offset_base + off_mem[n])), (n)&3)
				
		#define MATCH_CASE4(n) \
			MATCH_CASE(n+3); \
			MATCH_CASE(n+2); \
			MATCH_CASE(n+1); \
			MATCH_CASE(n); \
			cmp_idx = _mm512_permutex2var_epi8( \
				data0, \
				_mm512_add_epi8(_mm512_permutexvar_epi8(perm_idx_idx, compressed_idx), SPREAD_IDX), \
				data1 \
			); \
			perm_idx_idx = _mm512_add_epi8(perm_idx_idx, _mm512_set1_epi8(-8)); \
			cmp_match_pair = _mm_insert_epi64( \
				cmp_match_pair, _cvtmask64_u64(_mm512_cmpeq_epi8_mask(cmp_idx, cmp0)), ((n)>>2)&1 \
			)
		#define MATCH_CASE8(n) \
			MATCH_CASE4(n+4); \
			MATCH_CASE4(n); \
			cmp_matches = _mm512_inserti32x4(cmp_matches, cmp_match_pair, ((n)>>3) & 3)
		
		MATCH_CASE8(24);
		MATCH_CASE8(16);
		MATCH_CASE8(8);
		MATCH_CASE8(0);
		
		#undef MATCH_CASE8
		#undef MATCH_CASE4
		#undef MATCH_CASE
	}
	
	// get match lengths
	auto match_len = _mm512_maskz_cvtepi16_epi8(
		_cvtu32_mask32(_bzhi_u32(-1, num_match)),
		_mm512_popcnt_epi16(_mm512_andnot_si512(
			_mm512_add_epi16(cmp_matches, _mm512_set1_epi16(1)), cmp_matches
		))
	);
	
	auto long_matches = _mm256_test_epi8_mask(match_len, _mm256_set1_epi8(16));
	uint32_t long_matches_i = _cvtmask32_u32(long_matches);
	// TODO: consider extending lengths only after conflict resolution?
	if(long_matches_i != 0) {
		// find true match lengths, taking into consideration the buffer size
		compressed_offsets = _mm512_add_epi16(compressed_offsets, _mm512_set1_epi16(sizeof(__m128i)));
		compress_store_512_16(off_mem, long_matches, compressed_offsets);
		alignas(64) uint16_t idx_mem[32];
		auto compressed_idx2 = _mm512_add_epi16(compressed_idx, _mm512_set1_epi16(4 + sizeof(__m128i)));
		compress_store_512_16(idx_mem, long_matches, compressed_idx2);
		
		__mmask64 cmp_match;
		__m128i match_len_tmp, match_len_tmp1 = _mm_setzero_si128(), match_len_tmp2 = _mm_setzero_si128();
		__m256i match_len_long;
		for(int i=0; i<4; i++) {
			// TODO: last iteration can be simplified a bit
			if(HEDLEY_UNLIKELY(avail_len < 20 + (i+2)*sizeof(__m512i)))
				break; // TODO: handle partial loads
			
			cmp_match_pair = _mm_setzero_si128();
			cmp_matches = _mm512_setzero_si512();
			switch(_mm_popcnt_u32(long_matches_i)) { // might make more sense to quantize this
				default: HEDLEY_UNREACHABLE();
				#define MATCH_CASE(n) \
					HEDLEY_FALL_THROUGH; \
					case (n)+1: \
						cmp_match = _mm512_cmpeq_epi8_mask( \
							_mm512_loadu_si512(idx_base + idx_mem[n]), \
							_mm512_loadu_si512(offset_base + off_mem[n]) \
						); \
						cmp_match_pair = _mm_insert_epi64(cmp_match_pair, _cvtmask64_u64(cmp_match), (n)&1)
				#define MATCH_CASE2(n) \
					MATCH_CASE(n+1); \
					MATCH_CASE(n); \
					cmp_matches = _mm512_inserti32x4(cmp_matches, cmp_match_pair, ((n)/2)&3); \
					cmp_match_pair = _mm_setzero_si128()
				#define MATCH_CASE8(n) \
					MATCH_CASE2(n+6); \
					MATCH_CASE2(n+4); \
					MATCH_CASE2(n+2); \
					MATCH_CASE2(n); \
					match_len_tmp = _mm512_cvtepi64_epi8(_mm512_popcnt_epi64(_mm512_andnot_si512( \
						_mm512_add_epi64(cmp_matches, _mm512_set1_epi64(1)), cmp_matches \
					)))
				
				MATCH_CASE8(24);
				match_len_tmp2 = match_len_tmp;
				MATCH_CASE8(16);
				match_len_tmp2 = _mm_unpacklo_epi64(match_len_tmp, match_len_tmp2);
				MATCH_CASE8(8);
				match_len_tmp1 = match_len_tmp;
				MATCH_CASE8(0);
				match_len_tmp1 = _mm_unpacklo_epi64(match_len_tmp, match_len_tmp1);
				match_len_long = _mm256_inserti128_si256(_mm256_castsi128_si256(match_len_tmp1), match_len_tmp2, 1);
				
				#undef MATCH_CASE8
				#undef MATCH_CASE2
				#undef MATCH_CASE
			}
			
			match_len_long = _mm256_maskz_expand_epi8(long_matches, match_len_long);
			match_len = _mm256_adds_epu8(match_len, match_len_long);
			
			
			long_matches = _mm256_test_epi8_mask(match_len_long, _mm256_set1_epi8(64));
			long_matches_i = _cvtmask32_u32(long_matches);
			if(HEDLEY_LIKELY(long_matches_i == 0)) break;
			
			compressed_offsets = _mm512_add_epi16(compressed_offsets, _mm512_set1_epi16(sizeof(__m512i)));
			compress_store_512_16(off_mem, long_matches, compressed_offsets);
			compressed_idx2 = _mm512_add_epi16(compressed_idx2, _mm512_set1_epi16(sizeof(__m512i)));
			compress_store_512_16(idx_mem, long_matches, compressed_idx2);
		}
		// TODO: perhaps do nice len handling here?
	}
	
	
	// determine actual offsets
	assert(idx_base - offset_base <= (1<<WINDOW_ORDER));
	auto idx_offsets = _mm512_add_epi16(idx, _mm512_set1_epi16(idx_base-offset_base));
	// offsets need to be 0-based, so subtract an additional 1 from it
	// note that `idx_offsets - offsets - 1` == `idx_offsets + ~offsets`
	offsets = _mm512_add_epi16(
		idx_offsets,
		_mm512_ternarylogic_epi64(offsets, offsets, offsets, 0xf) // ~A
	);
	assert(_mm512_mask_cmpge_epu16_mask(match, offsets, _mm512_set1_epi16(1<<WINDOW_ORDER)) == 0);
	
	auto match_len_x = _mm256_maskz_expand_epi8(match, match_len);
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
	// compress down
	alignas(32) uint16_t mem_end[32];
	alignas(32) uint8_t mem_start[32];
	alignas(64) int16_t mem_val[32];
	auto match_offset = _mm512_add_epi16(_mm512_cvtepu8_epi16(match_indices), _mm512_set1_epi16(3));
	auto match_end = _mm512_add_epi16(_mm512_cvtepu8_epi16(matched_lengths_sub3), match_offset);
	compress_store_512_16(mem_end, match, match_end);
	compress_store_256_8(mem_start, match, match_indices);
	compress_store_512_16(mem_val, match, match_value);
	int num_match = _mm_popcnt_u32(_cvtmask32_u32(match));
	assert(num_match > 0);
	
	assert(mem_start[0] >= skip_til_index);
	int prev_i = 0;
	uint32_t selected = 0;
	for(int i=1; i<num_match; i++) {
		/*
		if(mem_end[prev_i] > mem_start[i]) {
			// conflict
			if(mem_start[i] - mem_start[prev_i] >= 4 && int(mem_end[i]) - int(mem_end[prev_i]) >= 4) {
				// shorten previous match and select it
				mem_end[prev_i] = mem_start[i];
				selected |= 1 << prev_i;
			}
			else if(mem_val[i] - mem_val[prev_i] <= 1) // slightly bias towards earlier match, as it's less likely to conflict
				continue; // discard this match
		} else {
			selected |= 1 << prev_i;
		}
		prev_i = i;
		*/
		
		lz77_cmov(mem_end[prev_i], (
			mem_end[prev_i] > mem_start[i]
			&& mem_start[i] - mem_start[prev_i] >= 4
			&& int(mem_end[i]) - int(mem_end[prev_i]) >= 4
		), uint16_t(mem_start[i]));
		
		int not_conflict = mem_end[prev_i] <= mem_start[i];
		selected |= not_conflict << prev_i; // if current doesn't conflict with previous, select previous
		lz77_cmov(prev_i, not_conflict || mem_val[i] - mem_val[prev_i] > 1, i); // if current match looks promising, set is as the next candidate
	}
	selected |= 1 << prev_i;
	skip_til_index = mem_end[prev_i];
	
	match_end = _mm512_maskz_expandloadu_epi16(match, mem_end);
	match_end = _mm512_sub_epi16(match_end, match_offset);
	matched_lengths_sub3 = _mm512_cvtepi16_epi8(match_end);
	
	return _pdep_u32(selected, _cvtmask32_u32(match));
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
			matched_lengths_sub4 = _mm256_castsi128_si256(lz77_get_match_len(indices, data, data2, offsets, search_base + search_base_offset, search_base, match1, match2, avail_len, match_offs, match_val));
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
			__m256i match_offs, match_val;
			matched_lengths_sub4 = _mm256_castsi128_si256(lz77_get_match_len(indices1, data, data2, offsets1, search_base + search_base_offset, search_base, match1, match2, avail_len, match_offs, match_val));
			matched_offsets = _mm512_castsi256_si512(match_offs);
			match_value = _mm512_castsi256_si512(match_val);
			match = _kor_mask16(match1, match2);
			
			if(_kortestz_mask16_u8(match3, match4) == 0) {
				matched_lengths_sub4 = _mm256_inserti128_si256(matched_lengths_sub4, lz77_get_match_len(indices2, data, data2, offsets2, search_base + search_base_offset, search_base, match3, match4, avail_len, match_offs, match_val), 1);
				matched_offsets = _mm512_inserti64x4(matched_offsets, match_offs, 1);
				match_value = _mm512_inserti64x4(match_value, match_val, 1);
				match = _mm512_kunpackw(_kor_mask16(match3, match4), match);
			}
		} else if(_kortestz_mask16_u8(match3, match4) == 0) {
			__m256i match_offs, match_val;
			matched_lengths_sub4 = _mm256_castsi128_si256(lz77_get_match_len(indices2, data, data2, offsets2, search_base + search_base_offset, search_base, match3, match4, avail_len, match_offs, match_val));
			matched_offsets = _mm512_castsi256_si512(match_offs);
			match_value = _mm512_castsi256_si512(match_val);
			match = _kor_mask16(match3, match4);
			// because the bottom 16 are effectively skipped, we need to remove these
			bounds = _pdep_u64(0xffff0000, bounds);
			match_indices = _mm256_castsi128_si256(_mm256_extracti128_si256(match_indices, 1));
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
