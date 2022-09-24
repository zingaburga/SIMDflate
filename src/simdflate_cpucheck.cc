
#include "../include/simdflate_cpucheck.h"

#if (defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)) && (!defined(_MSC_VER) || _MSC_VER >= 1600)
# ifdef _MSC_VER
#  include <intrin.h>
#  define CPUID __cpuid
#  define CPUIDX __cpuidex
# else
#  include <cpuid.h>
#  include <immintrin.h> // for _xgetbv
#  define CPUID(flags, n) __cpuid(n, flags[0], flags[1], flags[2], flags[3])
#  define CPUIDX(flags, eax, ecx) __cpuid_count(eax, ecx, flags[0], flags[1], flags[2], flags[3])
# endif

int simdflate_cpu_support() {
	int flags[4];
	
	CPUID(flags, 1);
	if((flags[2] & 0x18800002) != 0x18800002) // POPCNT + OSXSAVE + AVX + PCLMUL
		return SIMDFLATE_CPU_UNSUPPORTED;
	
	CPUID(flags, 0x80000001);
	if(!(flags[2] & 0x20)) // ABM
		return SIMDFLATE_CPU_UNSUPPORTED;
	
	auto xcr = _xgetbv(0); // 0 == _XCR_XFEATURE_ENABLED_MASK (constant not available on all compilers)
	if((xcr & 0xE6) != 0xE6) // AVX/AVX512 enabled
		return SIMDFLATE_CPU_UNSUPPORTED;
	
	CPUIDX(flags, 7, 0);
	if((flags[1] & 0xC0210108) == 0xC0210108 // AVX512BW + AVX512VL + AVX512IFMA + AVX512F + BMI2 + BMI1
		// change above to 0xD0010108 to add AVX512CD
	&& (flags[2] & 0x4D42) == 0x4D42) // AVX512VPOPCNTDQ + AVX512VNNI + VPCLMULQDQ + GFNI + AVX512VBMI2 + AVX512VBMI
		return SIMDFLATE_CPU_SUPPORTED;
	
	return SIMDFLATE_CPU_UNSUPPORTED;
}

#else
int simdflate_cpu_supported() { return SIMDFLATE_CPU_UNSUPPORTED; }
#endif
