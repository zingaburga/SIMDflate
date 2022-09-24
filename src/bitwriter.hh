#ifndef __SIMDFLATE_BITWRITER_H
#define __SIMDFLATE_BITWRITER_H

#include "common.hh"

// returns ((data << num_bits) | byte)
static HEDLEY_ALWAYS_INLINE __m512i shift_bits_into_vec(__m512i data, uint8_t byte, int num_bits) {
	assert(num_bits >= 0 && num_bits < 8);
	
	auto shift_in = _mm_insert_epi8(_mm_setzero_si128(), byte << (8-num_bits), 7);
	auto shifted_data = _mm512_mask_alignr_epi64(
		_mm512_castsi128_si512(shift_in),
		MASK8(_cvtu32)(0b11111110),
		data, data, 7
	);
	return _mm512_shldv_epi64(data, shifted_data, _mm512_set1_epi64(num_bits));
}

// Class to manage writing unaligned bits to a destination
// NOTE: class writes 8/64 bytes at a time, so assumes there is sufficient space in output for this
class BitWriter {
	unsigned bitpos; // number of bits written in last byte [0-7]
	uint8_t* output; // pointer to current byte, with bitpos bits written in it
public:
	BitWriter(void* out) : bitpos(0) {
		output = static_cast<uint8_t*>(out);
	}
	// NOTE: the Write* functions assume the underlying memory is zeroed before being written to; use ZeroWrite* functions if the assumption isn't true
	
	// write a 512-bit vector to the output stream; if we know len <= 505, a shortcut can be taken
	void ZeroWrite505(__m512i data, unsigned len) {
		assert(len <= 512-7);
		data = shift_bits_into_vec(data, *output, bitpos);
		_mm512_storeu_si512(output, data);
		bitpos += len;
		output += bitpos >> 3;
		bitpos &= 7;
	}
	void ZeroWrite512(__m512i data, unsigned len) {
		assert(len <= 512);
		auto len_shifted = len + bitpos;
		auto shift_in = _mm_insert_epi8(_mm_setzero_si128(), *output << (8-bitpos), 7);
		auto shifted_data = _mm512_mask_alignr_epi64(
			_mm512_castsi128_si512(shift_in),
			MASK8(_cvtu32)(0b11111110),
			data, data, 7
		);
		if(HEDLEY_UNLIKELY(len_shifted > 512)) {
			output[64] = _mm_extract_epi8(_mm512_extracti32x4_epi32(data, 3), 15) >> (8-bitpos);
		}
		data = _mm512_shldv_epi64(data, shifted_data, _mm512_set1_epi64(bitpos));
		_mm512_storeu_si512(output, data);
		output += len_shifted >> 3;
		bitpos = len_shifted & 7;
	}
	
	void Write64(uint64_t data, unsigned len) {
		assert(len == 64 || (data >> len) == 0);
		// hope this sequence compiles to `OR [mem], data<<bitpos`
		uint64_t o;
		memcpy(&o, output, sizeof(o));
		assert((o >> bitpos) == 0);
		o |= data << bitpos;
		memcpy(output, &o, sizeof(o));
		
		auto len_shifted = len + bitpos;
		if(HEDLEY_UNLIKELY(len_shifted > 64)) {
			output[8] = data >> (64 - bitpos);
		}
		output += len_shifted >> 3;
		bitpos = len_shifted & 7;
	}
	void Write57(uint64_t data, unsigned len) {
		assert(len <= 64-7);
		assert((data >> len) == 0);
		
		uint64_t o;
		memcpy(&o, output, sizeof(o));
		assert((o >> bitpos) == 0);
		o |= data << bitpos;
		memcpy(output, &o, sizeof(o));
		bitpos += len;
		output += bitpos >> 3;
		bitpos &= 7;
	}
	// the difference this function has from the above is that, since we don't know the underlying data is zeroed, we need to explicitly zero it
	void ZeroWrite57(uint64_t data, unsigned len) {
		assert(len <= 64-7);
		assert((data >> len) == 0);
		uint32_t o;
		memcpy(&o, output, sizeof(o));
		data = (data << bitpos) | _bzhi_u32(o, bitpos);
		memcpy(output, &data, sizeof(data));
		bitpos += len;
		output += bitpos >> 3;
		bitpos &= 7;
	}
	// if we assume the underlying stream is zeroed, this is equivalent to writing zero bits
	inline void Skip(unsigned len) {
		bitpos += len;
		output += bitpos >> 3;
		bitpos &= 7;
	}
	inline void SkipBytes(unsigned len) {
		output += len;
	}
	inline size_t Length(const void* start) const {
		return size_t(output - static_cast<const uint8_t*>(start) + (bitpos > 0));
	}
	inline uint64_t BitLength(const void* start) const {
		return uint64_t(output - static_cast<const uint8_t*>(start)) * 8 + bitpos;
	}
};


#endif
