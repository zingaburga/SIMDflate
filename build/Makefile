SRCDIR=../src
INCDIR=../include
CXXFLAGS=-O3 -g -DNDEBUG -Wall -Wextra -pedantic -std=c++14
#CXXFLAGS=-O0 -g -fsanitize=address,undefined -Wall -Wextra -pedantic -std=c++14
#LDFLAGS=-static-libasan

simdflate: simdflate.o simdflate_cpucheck.o ../main.cc
	$(CXX) -o simdflate $(CXXFLAGS) $(LDFLAGS) simdflate.o simdflate_cpucheck.o ../main.cc

simdflate.o: $(SRCDIR)/simdflate.cc $(INCDIR)/simdflate.h $(SRCDIR)/hedley.h $(SRCDIR)/bitwriter.hh $(SRCDIR)/checksum.hh $(SRCDIR)/common.hh $(SRCDIR)/histcount.hh $(SRCDIR)/huffmantree.hh $(SRCDIR)/huffmanenc.hh $(SRCDIR)/lz77.hh $(SRCDIR)/lz77data.hh
	$(CXX) -c -o simdflate.o -mavx512vl -mavx512bw -mavx512vbmi -mavx512vbmi2 -mavx512vnni -mavx512vpopcntdq -mavx512bitalg -mvpclmulqdq -mavx512ifma -mavx512cd -mgfni -mpclmul -mlzcnt -mbmi -mbmi2 $(CXXFLAGS) $(SRCDIR)/simdflate.cc

simdflate_cpucheck.o: $(SRCDIR)/simdflate_cpucheck.cc $(INCDIR)/simdflate_cpucheck.h
	$(CXX) -c -o simdflate_cpucheck.o -mxsave $(CXXFLAGS) $(SRCDIR)/simdflate_cpucheck.cc

clean:
	rm -f simdflate simdflate.o
