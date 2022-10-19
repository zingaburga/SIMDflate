#ifndef __SIMDFLATE_COMMON_H
#define __SIMDFLATE_COMMON_H

#include "hedley.h"

#if HEDLEY_GCC_VERSION_CHECK(12, 0, 0)
// workaround https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105593
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
# pragma GCC diagnostic ignored "-Wuninitialized"
# include <immintrin.h>
# pragma GCC diagnostic pop
#else
# include <immintrin.h> // x86 intrinsics
#endif

#include <stdint.h> // uintX_t etc
#include <cstring>  // memcpy
#include <cassert>  // assert

#include <algorithm> // std::min/max etc


// how large the LZ77 index table should be; each table entry holds 2*16-bit references
// 12 means 2^12 = 4096 entries, so 4B * 4096 = 16KB in size
const size_t MATCH_TABLE_ORDER = 12;
// size of LZ77 sliding window
#define WINDOW_ORDER 14 // = 16KB; max LZ77 window is 32KB (order=15)
// note that lz77_get_match_len may currently behave incorrectly when WINDOW_ORDER==15, due to overflowing a 16-bit signed int

// size of LZ77 output buffer; once this is filled, a block will be written
// note that histogramming uses 2x16-bit counters, meaning this value can't exceed 128KB (until sampling is introduced), but it's recommended to not exceed 64KB too much (reduced sampling accuracy due to using 16-bit counters)
// should also be larger than 128 bytes
const size_t OUTPUT_BUFFER_SIZE = 64*1024;



const unsigned NUM_SYM_TOTAL = 320; // 286 litlen symbols + 30 distance symbols, rounded up to 64

// some older compilers (GCC 9 / MSVC < 2017) lack vector zero extension intrinsics, so emulate them as appropriate
#if (defined(__clang__) && __clang_major__ >= 5 && \
     (!defined(__APPLE__) || __clang_major__ >= 7)) || \
    (defined(__GNUC__) && __GNUC__ >= 10) || \
    (defined(_MSC_VER) && _MSC_VER >= 1910)
#define ZEXT128_512 _mm512_zextsi128_si512
#define ZEXT128_256 _mm256_zextsi128_si256
#define ZEXT256_512 _mm512_zextsi256_si512
#else
#ifdef __OPTIMIZE__
// these are technically wrong, but likely good enough if optimizing as there's little reason for the compiler to put anything there but zeroes
#define ZEXT128_512 _mm512_castsi128_si512
#define ZEXT128_256 _mm256_castsi128_si256
#define ZEXT256_512 _mm512_castsi256_si512
#else
#define ZEXT128_512(x) _mm512_inserti32x4(_mm512_setzero_si512(), x, 0)
#define ZEXT128_256(x) _mm256_inserti128_si256(_mm256_setzero_si256(), x, 0)
#define ZEXT256_512(x) _mm512_inserti64x4(_mm512_setzero_si512(), x, 0)
#endif
#endif


// many *_mask8 intrinsics require AVX-512 DQ, but that extension doesn't really provide any benefit to us, so we'll just use _mask16 instead and ignore the unused bits
#define MASK8(f) f##_mask16
#define MASK8_TEST(f) f##_mask32_u8
#define MASK16_TEST(f) f##_mask32_u8

// a number of operations should be done on alternating elements
const __mmask64 MASK_ALTERNATE = 0x5555555555555555; // i.e. 0b01010101...


// shortcuts for constructing vector constants
template<typename fn>
static HEDLEY_ALWAYS_INLINE __m512i make_vec512_epi8(fn&& f) {
	return _mm512_set_epi8(
		f(63), f(62), f(61), f(60), f(59), f(58), f(57), f(56), f(55), f(54), f(53), f(52), f(51), f(50), f(49), f(48), f(47), f(46), f(45), f(44), f(43), f(42), f(41), f(40), f(39), f(38), f(37), f(36), f(35), f(34), f(33), f(32), f(31), f(30), f(29), f(28), f(27), f(26), f(25), f(24), f(23), f(22), f(21), f(20), f(19), f(18), f(17), f(16), f(15), f(14), f(13), f(12), f(11), f(10), f(9), f(8), f(7), f(6), f(5), f(4), f(3), f(2), f(1), f(0)
	);
}
#define VEC512_8(...) make_vec512_epi8([&](int _x){ return __VA_ARGS__; })
template<typename fn>
static HEDLEY_ALWAYS_INLINE __m512i make_vec512_epi16(fn&& f) {
	return _mm512_set_epi16(
		f(31), f(30), f(29), f(28), f(27), f(26), f(25), f(24), f(23), f(22), f(21), f(20), f(19), f(18), f(17), f(16), f(15), f(14), f(13), f(12), f(11), f(10), f(9), f(8), f(7), f(6), f(5), f(4), f(3), f(2), f(1), f(0)
	);
}
#define VEC512_16(...) make_vec512_epi16([&](int _x){ return __VA_ARGS__; })


// shortcut for looping over an array using 512-bit vectors, masking appropriately for the tail
template<typename fn_main, typename fn_end>
static HEDLEY_ALWAYS_INLINE void loop_u8x64(size_t size, const void* data, __m512i invalid_data, fn_main&& fm, fn_end&& fe) {
	auto data_ = static_cast<const uint8_t*>(data);
	size_t i = 0;
	for(; i+sizeof(__m512i)<=size; i+=sizeof(__m512i)) {
		fm(_mm512_loadu_si512(data_ + i), i);
	}
	if(size - i) { // tail vector
		uint64_t mask = _bzhi_u64(-1LL, size - i);
		fe(_mm512_mask_loadu_epi8(invalid_data, _cvtu64_mask64(mask), data_ + i), mask, i, size - i);
	}
}
template<typename fn>
static HEDLEY_ALWAYS_INLINE void loop_u8x64(size_t size, const void* data, __m512i invalid_data, fn&& f) {
	loop_u8x64(size, data, invalid_data, [&f](__m512i vdata, size_t& i) {
		f(vdata, _cvtu64_mask64(-1LL), i);
	}, [&f](__m512i vdata, __mmask64 mask, size_t& i, size_t) {
		f(vdata, mask, i);
	});
}
template<typename fn>
static HEDLEY_ALWAYS_INLINE void loop_u8x64(size_t size, const void* data, fn&& f) {
	loop_u8x64(size, data, _mm512_setzero_si512(), f);
}


// alternatives to _mm*_mask_compressstoreu_epi* which doesn't massively slow down Zen4, due to `VPCOMPRESS* [mem]` being uCode
// unlike the intrinsic, this does a full vector write instead of a masked one, but the caller should assume the contents of unmasked elements to be undefined (as we might optimise for Intel later by using the native intrinsic)
// likely only marginally less efficient on Intel
static HEDLEY_ALWAYS_INLINE void compress_store_512_8(void* dest, __mmask64 mask, __m512i data) {
	_mm512_storeu_si512(dest, _mm512_maskz_compress_epi8(mask, data));
}
static HEDLEY_ALWAYS_INLINE void compress_store_256_8(void* dest, __mmask32 mask, __m256i data) {
	_mm256_storeu_si256(static_cast<__m256i*>(dest), _mm256_maskz_compress_epi8(mask, data));
}
static HEDLEY_ALWAYS_INLINE void compress_store_512_16(void* dest, __mmask32 mask, __m512i data) {
	_mm512_storeu_si512(dest, _mm512_maskz_compress_epi16(mask, data));
}
static HEDLEY_ALWAYS_INLINE void compress_store_512_32(void* dest, __mmask16 mask, __m512i data) {
	_mm512_storeu_si512(dest, _mm512_maskz_compress_epi32(mask, data));
}


// for generating compile-time lookup tables
// from https://joelfilho.com/blog/2020/compile_time_lookup_tables_in_cpp/
#include <array>
#include <cstddef>
#include <utility>
template <std::size_t Length, typename Generator, std::size_t... Indexes>
constexpr auto lut_impl(Generator &&f, std::index_sequence<Indexes...>) {
	using content_type = decltype(f(std::size_t{0}));
	return std::array<content_type, Length>{{f(Indexes)...}};
}
template <std::size_t Length, typename Generator>
constexpr auto lut(Generator &&f) {
	return lut_impl<Length>(std::forward<Generator>(f), std::make_index_sequence<Length>{});
}


#endif
