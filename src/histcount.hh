#ifndef __SIMDFLATE_HISTCOUNT_H
#define __SIMDFLATE_HISTCOUNT_H

#include "common.hh"
#include "lz77data.hh"

// histogram LZ77 symbols, returns whether all symbols were histogrammed (false) or a sample was used (true)
bool symbol_histogram(uint16_t* sym_counts, const Lz77Data& lz77output) {
	// additional histogram bin; used to reduce dependency chain on having only one bin
	alignas(64) uint16_t count_xbin[NUM_SYM_TOTAL] = {};
	bool is_sample = false; // TODO: determine if sampling
	// WARNING: enabling sampling currently allows exceeding the max deflate length; may be possible to enable it by changing the max length calculation to include a buffer or similar at the end?
	
	// if we're taking only a sample, we can't exclude any symbols, so need to set minimum count of all to 1
	auto count_fill = is_sample ? _mm512_set1_epi16(1) : _mm512_setzero_si512();
	for(unsigned i=0; i<NUM_SYM_TOTAL; i+=32) {
		_mm512_store_si512(sym_counts + i, count_fill);
	}
	
	// histogram 8x u16 into bin
	auto count_to_bin = [](__m128i data, uint16_t* bin) {
		// TODO: see if 32b extract is any better
		
		// use 8-bit pointer, but typecast to 16-bit to avoid *2 during address calculation
		uint8_t* bin8 = reinterpret_cast<uint8_t*>(bin);
		uint64_t bottom = _mm_cvtsi128_si64(data);
		(*reinterpret_cast<uint16_t*>(bin8 + _mm_extract_epi16(data, 4)))++;
		(*reinterpret_cast<uint16_t*>(bin8 + (bottom & 0xffff)))++;
		bottom >>= 16;
		(*reinterpret_cast<uint16_t*>(bin8 + _mm_extract_epi16(data, 5)))++;
		(*reinterpret_cast<uint16_t*>(bin8 + (bottom & 0xffff)))++;
		bottom >>= 16;
		(*reinterpret_cast<uint16_t*>(bin8 + _mm_extract_epi16(data, 6)))++;
		(*reinterpret_cast<uint16_t*>(bin8 + (bottom & 0xffff)))++;
		bottom >>= 16;
		(*reinterpret_cast<uint16_t*>(bin8 + _mm_extract_epi16(data, 7)))++;
		(*reinterpret_cast<uint16_t*>(bin8 + bottom))++;
	};
	
	assert(is_sample || lz77output.len < 131072); // limit of 2x16b counters
	for(size_t i=0; i<lz77output.len; i+=sizeof(__m512i)) { // includes tail vector (data is padded)
		auto data = _mm512_load_si512(lz77output.data + i);
		auto is_lendist = _cvtu64_mask64(lz77output.is_lendist[i/sizeof(__m512i)]);
		// remove top bit of len/dist symbols, whilst zeroing out extra bits
		data = _mm512_mask_subs_epu8(data, is_lendist, data, _mm512_set1_epi8(-128));
		
		// since len/dist symbols start at 256, insert a '1' in the upper byte when converting symbols from 8-bit to 16-bit
		// note that extra-bits symbols get mapped exactly to 256, which we'll fix up at the end
		auto lendist_bit = _mm512_maskz_mov_epi8(is_lendist, _mm512_set1_epi8(1));
		auto data0 = _mm512_unpacklo_epi8(data, lendist_bit);
		auto data1 = _mm512_unpackhi_epi8(data, lendist_bit);
		
		// symbols should now be guaranteed to be between 0-317, and not 286 or 287
		assert(_mm512_cmpgt_epu16_mask(data0, _mm512_set1_epi16(317)) == 0);
		assert(_mm512_cmpgt_epu16_mask(data1, _mm512_set1_epi16(317)) == 0);
		assert(_mm512_cmpeq_epu16_mask(data0, _mm512_set1_epi16(286)) == 0 && _mm512_cmpeq_epu16_mask(data0, _mm512_set1_epi16(287)) == 0);
		assert(_mm512_cmpeq_epu16_mask(data1, _mm512_set1_epi16(286)) == 0 && _mm512_cmpeq_epu16_mask(data1, _mm512_set1_epi16(287)) == 0);
		
		// double indices to simplify addressing during histogramming
		data0 = _mm512_add_epi16(data0, data0);
		data1 = _mm512_add_epi16(data1, data1);
		
		// do actual counting
		count_to_bin(_mm512_castsi512_si128(data0), sym_counts);
		count_to_bin(_mm512_castsi512_si128(data1), count_xbin);
		count_to_bin(_mm256_extracti128_si256(_mm512_castsi512_si256(data0), 1), sym_counts);
		count_to_bin(_mm256_extracti128_si256(_mm512_castsi512_si256(data1), 1), count_xbin);
		auto data0_top = _mm512_extracti64x4_epi64(data0, 1);
		auto data1_top = _mm512_extracti64x4_epi64(data1, 1);
		count_to_bin(_mm256_castsi256_si128(data0_top), sym_counts);
		count_to_bin(_mm256_castsi256_si128(data1_top), count_xbin);
		count_to_bin(_mm256_extracti128_si256(data0_top, 1), sym_counts);
		count_to_bin(_mm256_extracti128_si256(data1_top, 1), count_xbin);
	}
	
	// aggregate additional histogram bins
	if(0 && !is_sample && lz77output.len >= 65536) { // if exceeding 16b limit, aggregate average to stay within 16b; TODO: also consider being a bit smarter and only do this if we'd exceed the limit?
		for(unsigned i=0; i<NUM_SYM_TOTAL; i+=32) {
			_mm512_store_si512(sym_counts + i, _mm512_avg_epu16(
				_mm512_load_si512(sym_counts + i),
				_mm512_load_si512(count_xbin + i)
			));
		}
	} else {
		// we don't care too much if we exceed 16-bit limit, as it'll mean we have a very skewed distribution
		for(unsigned i=0; i<NUM_SYM_TOTAL; i+=32) {
			_mm512_store_si512(sym_counts + i, _mm512_adds_epu16(
				_mm512_load_si512(sym_counts + i),
				_mm512_load_si512(count_xbin + i)
			));
		}
	}
	sym_counts[256] = 1; // clear out miscounted extra-bits symbols, whilst correctly setting the end-of-block count
	return is_sample;
}

#endif
