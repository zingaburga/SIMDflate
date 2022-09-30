#ifndef __SIMDFLATE_H
#define __SIMDFLATE_H

#include <stdlib.h> // size_t

#ifdef __cplusplus
extern "C" {
#endif

// compress using raw DEFLATE, or with zlib/gzip wrappers
size_t simdflate_deflate(void* dest, const void* src, size_t len);
size_t simdflate_zlib(void* dest, const void* src, size_t len);
size_t simdflate_gzip(void* dest, const void* src, size_t len);

size_t simdflate_max_deflate_len(size_t len);
inline size_t simdflate_max_zlib_len(size_t len) {
	return simdflate_max_deflate_len(len) + 6;
}
inline size_t simdflate_max_gzip_len(size_t len) {
	return simdflate_max_deflate_len(len) + 18;
}

// (boolean) whether SIMDflate was actually compiled; if the compiler doesn't support SIMDflate, this will be set to 0
extern const int SIMDFLATE_COMPILER_SUPPORTED;

#ifdef __cplusplus
}
#endif
#endif
