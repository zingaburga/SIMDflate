#ifndef __SIMDFLATE_CPUCHECK_H
#define __SIMDFLATE_CPUCHECK_H

#ifdef __cplusplus
extern "C" {
#endif

// determines wether the current CPU supports instructions required for SIMDflate to run
enum {
	SIMDFLATE_CPU_UNSUPPORTED = 0,
	SIMDFLATE_CPU_SUPPORTED = 0x100
};
int simdflate_cpu_support();

// checks whether SIMDflate can be used, by checking that it's actually compiled, and the CPU is supported
#define SIMDFLATE_AVAILABLE (SIMDFLATE_COMPILER_SUPPORTED && simdflate_cpu_support())

#ifdef __cplusplus
}
#endif
#endif
