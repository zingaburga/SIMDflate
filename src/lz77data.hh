#ifndef __SIMDFLATE_LZ77DATA_H
#define __SIMDFLATE_LZ77DATA_H

#include "common.hh"

// structure to hold LZ77 encoded data
/*
  LZ77 data is encoded as follows:
  - is_lendist is an array of bits, each mapping to a byte in `data`
    * If the bit is 0, the corresponding `data` byte is a literal symbol (0-255).
    * If the bit is 1, the corresponding `data` byte is either a length symbol (257-285, mapped to 129-157), a distance symbol (0-29 => 160-189) or up to 7 of the least significant extra bits for the preceeding length/distance symbol (0-127)
  - as somewhat implied above, distance symbols are mapped into the literal/length alphabet.  The latter covers the range 0-285, whilst the distance symbols are mapped to 288-317
  - since `data` is only using 8-bit units, you could consider is_lendist effectly holding the 9th bit, but for length/distance symbols, the top bit of `data` is set
  - xbits_hi holds extra-bits data for distance symbols with >7 extra bits
    * the most significant bits are stored here, which is to be concatenated with the low bits stored in `data`
    * xbits_hi is in a packed format to save memory, where a byte is only reserved for instances with >7 extra bits
  
  Example: where [n] = extra bits value, __d = distance symbol
    Actual symbols: 50, 260,  4d, [0], 285, 23d, [520]
  Encoded as:
              data: 50, 132, 164,   0, 157, 183,    8
        is_lendist:  0,   1,   1,   1,   1,   1,    1
          xbits_hi: 4
  Explanation:
  - first symbol is a literal (50), encoded as-is into `data`, and corresponding is_lendist bit set to 0
  - second symbol is a length=6 code: subtract 128 from it and store in `data` as 132, and set corresponding bit in is_lendist to 1
    - this length symbol doesn't have any extra bits, so just proceed
  - third symbol is a distance={5 or 6} code: add 160 to it and store in `data` as 164, and set corresponding bit in is_lendist to 1
    - this distance symbol does have 1 extra bit, of value 0, so store that 0 in `data`, set is_lendist bit to 1
    - since there's fewer than 7 extra bits here, nothing needs to be done with xbits_hi
  - the next symbol is a length=258 symbol; subtract 128 and store in `data` as 157, with is_lendist bit as 1
  - the next symbol is a distance={3073-4096} code: add 160 and store in `data` as 183, with is_lendist bit=1
    - this distance symbol has 10 extra bits, but we can only store up to 7 bits in data
    - as a result, we'll split the extra bits into two halves: the 7 least significant bits will go in `data`, whilst the remaining 3 bits will go to xbits_hi
    - the lower half's value is 8, whilst the upper half is 4, since 520 = (4<<7) + 8
    - write the lower half to `data`, and the upper half to xbits_hi
  
  Compared to the more normal layout of using 16-bit per symbol entries, this memory layout uses much less (cache) memory, as most symbols only consume 9 bits, and can be processed reasonably efficiently
 */
constexpr size_t LZ77DATA_XBITS_HI_SIZE = ((OUTPUT_BUFFER_SIZE +2) / 3) + sizeof(__m512i); // largest possible size is 1/3 of `data` (expected size is much smaller)
struct Lz77Data {
	size_t len, xbits_hi_len;
	// add padding onto the end of these, to not have to worry about masking when reading the data
	alignas(8)  uint64_t is_lendist[(OUTPUT_BUFFER_SIZE+63)/64 + 1];
	alignas(64) uint8_t data[OUTPUT_BUFFER_SIZE + sizeof(__m512i)];
	alignas(64) uint8_t xbits_hi[LZ77DATA_XBITS_HI_SIZE];
	
	inline Lz77Data() {
		// is_lendist is written to using BitWriter, so needs to be zeroed up front
		memset(is_lendist, 0, sizeof(is_lendist));
		len = 0;
	}
};

#endif
