#define _CRT_SECURE_NO_WARNINGS 1

#include "include/simdflate.h"
#include "include/simdflate_cpucheck.h"
#include <cstdio>
#include <cerrno>
#include <vector>
#include <string.h>
#include <chrono>

void print_usage(const char* bin) {
	fprintf(stderr, "Usage: %s [options] input output\n", bin);
	fprintf(stderr, "  -r<n>    Run <n> repetitions\n");
	fprintf(stderr, "  -z       Zlib format instead of gzip\n");
	fprintf(stderr, "  -d       Raw Deflate instead of gzip\n");
}

int main(int argc, char** argv) {
	fprintf(stderr, "SIMDflate support check: compiler=%s, CPU=%s\n", SIMDFLATE_COMPILER_SUPPORTED ?"yes":"no", simdflate_cpu_support() ?"yes":"no");
	if(argc < 3) {
		print_usage(argv[0]);
		return 1;
	}
	
	int reps = 1;
	int format = 0; // gzip
	char** argp = argv + 1;
	while(argc-- > 3) {
		if((*argp)[0] != '-') {
			print_usage(argv[0]);
			return 1;
		}
		switch((*argp)[1]) {
			case 'r':
				reps = atoi((*argp) + 2);
				break;
			case 'z': format = 1; break;
			case 'd': format = 2; break;
			default:
				print_usage(argv[0]);
				return 1;
		}
		argp++;
	}
	
	auto f = fopen(argp[0], "rb");
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
	
	float us;
	size_t output_len;
	std::vector<unsigned char> output(simdflate_max_gzip_len(data.size()));
	
	if(format == 0) {
		auto start = std::chrono::high_resolution_clock::now();
		output_len = simdflate_gzip(output.data(), data.data(), data.size());
		for(int i=1; i<reps; i++)
			simdflate_gzip(output.data(), data.data(), data.size());
		auto end = std::chrono::high_resolution_clock::now();
		
		us = float(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
	}
	else if(format == 1) {
		auto start = std::chrono::high_resolution_clock::now();
		output_len = simdflate_zlib(output.data(), data.data(), data.size());
		for(int i=1; i<reps; i++)
			simdflate_zlib(output.data(), data.data(), data.size());
		auto end = std::chrono::high_resolution_clock::now();
		
		us = float(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
	}
	else if(format == 2) {
		auto start = std::chrono::high_resolution_clock::now();
		output_len = simdflate_deflate(output.data(), data.data(), data.size());
		for(int i=1; i<reps; i++)
			simdflate_deflate(output.data(), data.data(), data.size());
		auto end = std::chrono::high_resolution_clock::now();
		
		us = float(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
	}
	
	auto input_len = double(data.size());
	printf("Size: %zu bytes (%.3f%%)\n", output_len, (double(output_len) / input_len) * 100);
	printf("Time: %.3f s (%.3f MiB/s)\n", us/1000000, input_len*reps/1.048576 / us);
	
	f = fopen(argp[1], "wb");
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
