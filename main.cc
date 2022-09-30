#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "include/simdflate.h"
#include "include/simdflate_cpucheck.h"
#include <cstdio>
#include <cerrno>
#include <vector>
#include <string.h>
#include <chrono>


int main(int argc, char** argv) {
	fprintf(stderr, "SIMDflate support check: compiler=%s, CPU=%s\n", SIMDFLATE_COMPILER_SUPPORTED ?"yes":"no", simdflate_cpu_support() ?"yes":"no");
	if(argc < 3) {
		fprintf(stderr, "Usage: %s input output\n", argv[0]);
		return 1;
	}
	
	int reps = 1;
	auto f = fopen(argv[1], "rb");
	if(!f) {
		fprintf(stderr, "Failed to open input file: %s\n", strerror(errno));
		return 1;
	}
	fseek(f, 0, SEEK_END);
	std::vector<unsigned char> data(ftell(f));
	fseek(f, 0, SEEK_SET);
	if(fread(data.data(), 1, data.size(), f) != data.size()) {
		fprintf(stderr, "Failed to read from input file: %s\n", strerror(errno));
		fclose(f);
		return 1;
	}
	fclose(f);
	
	std::vector<unsigned char> output(simdflate_max_gzip_len(data.size()));
	
	auto start = std::chrono::high_resolution_clock::now();
	auto output_len = simdflate_gzip(output.data(), data.data(), data.size());
	for(int i=1; i<reps; i++)
		simdflate_gzip(output.data(), data.data(), data.size());
	auto end = std::chrono::high_resolution_clock::now();
	
	auto input_len = double(data.size());
	auto us = float(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
	printf("Ratio: %.3f%%\n", (double(output_len) / input_len) * 100);
	printf("Speed: %.3f MiB/s\n", input_len*reps/1.048576 / us);
	
	f = fopen(argv[2], "wb");
	if(!f) {
		fprintf(stderr, "Failed to open output file: %s\n", strerror(errno));
		return 1;
	}
	if(fwrite(output.data(), 1, output_len, f) != output_len) {
		fprintf(stderr, "Failed to write to output file: %s\n", strerror(errno));
		fclose(f);
		return 1;
	}
	fclose(f);
	
	return 0;
}
