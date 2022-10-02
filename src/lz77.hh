
#include "common.hh"
#include "bitwriter.hh"
#include "lz77data.hh"


const uint32_t HASH_PROD = 0x9E3779B1; // 0x9E3779B1 = 2^32 * GoldenRatio
const size_t NICE_MATCH_LEN = 64; // max 64



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




// get match length between two addresses, checking at least NICE_MATCH_LEN bytes
static inline int get_match_len(const void* addr1, const void* addr2) {
	if(NICE_MATCH_LEN > 32) {
		auto cmp = _mm512_cmpeq_epu8_mask(_mm512_loadu_si512(addr1), _mm512_loadu_si512(addr2));
		return _tzcnt_u64(_cvtmask64_u64(cmp) + 1);
	} else if(NICE_MATCH_LEN > 16) {
		auto cmp = _mm256_cmpeq_epu8_mask(_mm256_loadu_si256(static_cast<const __m256i*>(addr1)), _mm256_loadu_si256(static_cast<const __m256i*>(addr2)));
		return _tzcnt_u32(~_cvtmask32_u32(cmp));
	} else {
		auto cmp = _mm_cmpeq_epu8_mask(_mm_loadu_si128(static_cast<const __m128i*>(addr1)), _mm_loadu_si128(static_cast<const __m128i*>(addr2)));
		return _tzcnt_u32(~_cvtmask16_u32(cmp));
	}
}

// returns symbol % 256
static inline uint8_t lz77_length_symbol(int length, unsigned& xbits_len) {
	length -= 3;
	assert(length >= 0 && length < 256);
	if(length < 8) {
		xbits_len = 0;
		return 1 + length;
	}
	if(length == 255) {
		xbits_len = 0;
		return 29;
	}
	auto len_log2 = 30 - _lzcnt_u32(length);
	xbits_len = len_log2 - 1;
	auto sym_extra = (length >> xbits_len) & 3;
	return 1 + len_log2 * 4 + sym_extra;
}
static inline uint8_t lz77_distance_symbol(int distance, unsigned& xbits_len) {
	distance--;
	assert(distance >= 0 && distance < 32768);
	if(distance < 4) {
		xbits_len = 0;
		return distance;
	}
	auto dist_log2 = 31^_lzcnt_u32(distance);
	xbits_len = dist_log2 - 1;
	auto sym_extra = (distance >> xbits_len) & 1;
	return dist_log2 * 2 + sym_extra;
}

static HEDLEY_ALWAYS_INLINE void lz77_find_twomatch(
	__m512i source_data, __m512i indices, uint32_t* match_offsets, uint16_t search_base_offset, const void* search_base,
	__mmask16& match1, __mmask16& match2, __m512i& wrapped_offsets
) {
	// compute hash
	// avoid VPMULLD as it is considered "AVX-512 heavy" (and generally slower than VPMADDWD)
	auto hash_data = _mm512_madd_epi16(source_data, _mm512_set1_epi32(HASH_PROD));
	hash_data = _mm512_and_si512(hash_data, _mm512_set1_epi32((1<<MATCH_TABLE_ORDER)-1));
	// TODO: see if above AND can be replaced with a shift
	
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
		// TODO: lzcnt may be a 'heavy' instruction
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
	auto hash_data = _mm512_madd_epi16(source_data, _mm512_set1_epi32(HASH_PROD));
	hash_data = _mm512_and_si512(hash_data, _mm512_set1_epi32((1<<MATCH_TABLE_ORDER)-1));
	
	auto current_offsets = _mm512_add_epi16(indices, _mm512_set1_epi32(search_base_offset));
	current_offsets = _mm512_slli_epi32(current_offsets, 16);
	_mm512_i32scatter_epi32(match_offsets, hash_data, current_offsets, sizeof(uint32_t));
	
	// TODO: consider in-vector matches
}

template<bool first_vec>
static unsigned lz77_vec(__m512i data, __m128i data2, uint32_t* match_offsets, const uint8_t* HEDLEY_RESTRICT search_base, uint16_t search_base_offset, unsigned& skip_til_index,
	uint8_t* HEDLEY_RESTRICT output, uint8_t* HEDLEY_RESTRICT& out_xbits_hi, BitWriter& is_lendist) {
	
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
	
	
	uint32_t match1_int, match2_int;
	alignas(64) int16_t offsets_mem[64];
	alignas(16) uint8_t indices_mem[32];
	
	// if boundary count <= 1/4
	// TODO: most text has 16-32 matches => maybe get rid of the <=16 case?
	if(num_bounds <= sizeof(__m512i) / 4) {
		// set some bits to reach the limit - this is because gather/scatter isn't any faster with masked out elements, so we may as well make full use of it
		// it also saves us from having to keep track of invalid elements
		bounds |= _pdep_u64(BITSET_TABLE[num_bounds], ~bounds);
		
		auto indices = _mm512_maskz_compress_epi8(_cvtu64_mask64(bounds), VEC512_8(_x));
		// expand indices 4x
		indices = _mm512_permutexvar_epi8(VEC512_8(_x/4), indices);
		// shuffle data into place
		auto source_data = _mm512_permutex2var_epi8(data, _mm512_add_epi8(indices, _mm512_set1_epi32(0x03020100)), _mm512_castsi128_si512(data2));
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
		}
		
		if(!first_vec && _kortestz_mask16_u8(match1, match2) == 0) {
			// TODO: see if this can be optimised better
			// - perhaps consider idea of doing a 64b gather & comparing iff not many matches
			_mm512_store_si512(offsets_mem, offsets);
			// _mm512_cvtepi32_storeu_epi8 is missing, but all compilers handle the following correctly
			_mm512_mask_cvtepi32_storeu_epi8(indices_mem, -1, indices);
			match1_int = _cvtmask16_u32(match1);
			match2_int = _cvtmask16_u32(match2);
		} else {
			// no match found, store and bail
			_mm512_mask_storeu_epi8(output - skip_til_index, _cvtu64_mask64(0xffffffffffffffffULL << skip_til_index), data);
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
		
		auto indices = _mm512_maskz_compress_epi8(_cvtu64_mask64(bounds), VEC512_8(_x));
		// expand indices 4x
		auto indices1 = _mm512_permutexvar_epi8(VEC512_8(_x/4), indices);
		auto indices2 = _mm512_permutexvar_epi8(VEC512_8(_x/4+16), indices);
		// shuffle data into place
		auto source_data1 = _mm512_permutexvar_epi8(_mm512_add_epi8(indices1, _mm512_set1_epi32(0x03020100)), data);
		auto source_data2 = _mm512_permutex2var_epi8(data, _mm512_add_epi8(indices2, _mm512_set1_epi32(0x03020100)), _mm512_castsi128_si512(data2));
		indices1 = _mm512_and_si512(indices1, _mm512_set1_epi32(0xff));
		indices2 = _mm512_and_si512(indices2, _mm512_set1_epi32(0xff));
		
		__mmask16 match1, match2, match3, match4;
		__mmask32 match13, match24;
		__m512i offsets1, offsets2;
		if(first_vec) {
			lz77_set_matches(source_data1, indices1, match_offsets, search_base_offset);
		} else {
			lz77_find_twomatch(
				source_data1, indices1, match_offsets, search_base_offset, search_base,
				match1, match2, offsets1
			);
		}
		lz77_find_twomatch(
			source_data2, indices2, match_offsets, search_base_offset, search_base,
			match3, match4, offsets2
		);
		
		if(!first_vec) {
			match13 = _mm512_kunpackw(match3, match1);
			match24 = _mm512_kunpackw(match4, match2);
		}
		
		if(!first_vec && _kortestz_mask32_u8(match13, match24) == 0) {
			_mm512_store_si512(offsets_mem, offsets1);
			_mm512_store_si512(offsets_mem + 32, offsets2);
			// _mm512_cvtepi32_storeu_epi8 is missing, but all compilers handle the following correctly
			_mm512_mask_cvtepi32_storeu_epi8(indices_mem, -1, indices1);
			_mm512_mask_cvtepi32_storeu_epi8(indices_mem + 16, -1, indices2);
			match1_int = _cvtmask32_u32(match13);
			match2_int = _cvtmask32_u32(match24);
		} else if(first_vec && _kortestz_mask16_u8(match3, match4) == 0) {
			_mm512_store_si512(offsets_mem, offsets2);
			_mm512_mask_cvtepi32_storeu_epi8(indices_mem, -1, indices2);
			match1_int = _cvtmask16_u32(match3);
			match2_int = _cvtmask16_u32(match4);
		} else {
			// no match found, store and bail
			_mm512_mask_storeu_epi8(output - skip_til_index, _cvtu64_mask64(0xffffffffffffffffULL << skip_til_index), data);
			auto bytes_written = sizeof(__m512i)-skip_til_index;
			skip_til_index = 0;
			is_lendist.Skip(bytes_written);
			return bytes_written;
		}
	}
	
	
	
	// TODO: change strat to do more checks
	// - ideas: backward searches, shortening match lengths for subsequent matches etc
	auto match_either = match1_int | match2_int;
	auto match_both = match1_int & match2_int;
	unsigned outpos = 0;
	while(match_either) {
		auto bit_idx = _tzcnt_u32(match_either);
		auto cur_idx = indices_mem[bit_idx];
		if(cur_idx >= skip_til_index) {
			auto cur_ptr = search_base + search_base_offset + cur_idx;
			int len, off;
			// TODO: since we know the first 4 bytes match, those can be skipped during compare
			if((match_both >> bit_idx) & 1) {
				auto off1 = offsets_mem[bit_idx*2];
				auto off2 = offsets_mem[bit_idx*2 + 1];
				auto len1 = get_match_len(cur_ptr, search_base + off1);
				auto len2 = get_match_len(cur_ptr, search_base + off2);
				if(len2 >= len1) { // prefer len2 as it's closer
					len = len2;
					off = off2;
				} else {
					len = len1;
					off = off1;
				}
			} else {
				off = offsets_mem[bit_idx*2 + ((match2_int >> bit_idx) & 1)];
				len = get_match_len(cur_ptr, search_base + off);
			}
			assert(len >= 4); // remove this if we skip checking the first 4 bytes
			
			// write literals
			is_lendist.Skip(cur_idx - skip_til_index);
			_mm512_mask_storeu_epi8(output + outpos - skip_til_index, (_bzhi_u64(0xffffffffffffffffULL << skip_til_index, cur_idx)), data);
			outpos += cur_idx - skip_til_index;
			// write len/off
			off = cur_ptr - (search_base + off);
			assert(off > 0);
			auto lendist_pos = outpos;
			unsigned len_xbits;
			output[lendist_pos++] = 128 + lz77_length_symbol(len, len_xbits);
			if(len_xbits > 0) {
				assert(len_xbits <= 5);
				output[lendist_pos++] = _bzhi_u32(len-3, len_xbits);
			}
			output[lendist_pos++] = 160 + lz77_distance_symbol(off, len_xbits);
			if(len_xbits > 0) {
				auto len_xbits_bytes = 1;
				uint16_t xbits = _bzhi_u32(off-1, len_xbits);
				assert(len_xbits <= WINDOW_ORDER-2);
#if WINDOW_ORDER > 9
				// xbits is split across output and xbits_hi
				output[lendist_pos++] = xbits & 0x7f;
				*out_xbits_hi = xbits >> 7;
				out_xbits_hi += (len_xbits >> 3);
#else
				output[lendist_pos++] = xbits;
#endif
			}
			is_lendist.Write57(_bzhi_u32(15, lendist_pos - outpos), lendist_pos - outpos);
			assert(lendist_pos - outpos <= 5 && lendist_pos - outpos >= 2);
			outpos = lendist_pos;
			
			// TODO: if len >= NICE_MATCH_LEN, need to scan for total length
			
			// skip subsequent matches that overlap
			// TODO: ...or shorten matches?
			skip_til_index = cur_idx + len;
		}
		
		match_either = _blsr_u32(match_either);
	}
	
	// write trailing literals
	if(skip_til_index < 64) {
		is_lendist.Skip(sizeof(__m512i) - skip_til_index);
		_mm512_mask_storeu_epi8(output + outpos - skip_til_index, _cvtu64_mask64(0xffffffffffffffffULL << skip_til_index), data);
		outpos += sizeof(__m512i) - skip_til_index;
		skip_til_index = 0;
	} else {
		skip_til_index -= 64;
	}
	return outpos;
}

template<class ChecksumClass>
static void lz77_encode(
	Lz77Data& output,
	// inputs (+gets changed)
	const void* HEDLEY_RESTRICT& src, size_t& len,
	uint16_t& window_offset, unsigned& skip_next_bytes,
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
			// for the first buffer, there's an issue with the first 4 bytes matching itself (since all hash table entries point to it)
			// we avoid this issue by avoiding trying to find matches for the first vector (which would be pointless to do anyway)
			auto data = _mm512_loadu_si512(_src);
			auto next_data = _mm_cvtsi32_si128(*reinterpret_cast<const uint32_t*>(_src + sizeof(__m512i)));
			
			output.len += lz77_vec<true>(data, next_data, match_offsets, _src, 0, skip_next_bytes,
				output.data + output.len, xbits_hi_ptr, is_lendist_writer);
			cksum.update(data);
			window_offset += sizeof(__m512i);
		}
		
		auto safe_end = reinterpret_cast<const uint8_t*>(uintptr_t(src_end - sizeof(__m512i)) & -int(sizeof(__m512i)));
		assert(_src < safe_end);
		while(_src < safe_end) {
			auto win_size = std::min(size_t(safe_end - _src), size_t(1<<WINDOW_ORDER));
			for(; window_offset < win_size && output.len < OUTPUT_BUFFER_SIZE - sizeof(__m512i)*2; window_offset += sizeof(__m512i)) {
				auto data = _mm512_loadu_si512(_src + window_offset);
				auto next_data = _mm_cvtsi32_si128(*reinterpret_cast<const uint32_t*>(_src + window_offset + sizeof(__m512i)));
				
				output.len += lz77_vec<false>(data, next_data, match_offsets, _src, window_offset, skip_next_bytes,
					output.data + output.len, xbits_hi_ptr, is_lendist_writer);
				cksum.update(data);
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
