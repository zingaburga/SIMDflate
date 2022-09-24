#ifndef __SIMDFLATE_LZ77DATA_H
#define __SIMDFLATE_LZ77DATA_H

#include "common.hh"

// structure to hold LZ77 encoded data
/*
  LZ77 data is encoded as follows:
  - is_lendist is an array of bits, each mapping to a byte in `data`
    * If the bit is 0, the corresponding `data` byte is a literal symbol (0-255).
    * If the bit is 1, the corresponding `data` byte is a length symbol (257-285, mapped to 129-157), a distance symbol (0-29 => 160-189) or _number_ of extra bits for the preceeding length/distsance symbol (1-7)
  - as suggested above, distance symbols are mapped into the literal/length alphabet.  The latter covers the range 0-285, whilst the distance symbols are mapped to 288-317
  - since `data` is only using 8-bit units, is_lendist effectly holds the 9th bit, but for length/distance symbols, the top bit of `data` is set
  - xbits holds the actual extra-bits data (remember that `data` only holds the lengths of the extra bits)
    * xbits is in a packed format to save memory, where each byte corresponds with a extra-bits length byte in `data`
    * if there are more than 7 extra bits (for distance symbols) it's stored as two bytes in `data` and xbits, where the length in `data` maxes out at 7
  
  Example: where [n] = extra bits value, {m} = distance symbol
    Actual symbols: 50, 260, {4}, [0], 285, {23}, [520]
  Encoded as:
              data: 50, 132, 164,   1, 157,  183, 7, 3
        is_lendist:  0,   1,   1,   1,   1,    1, 1, 1
             xbits: 0, 8, 4
  Explanation:
  - first symbol is a literal (50), encoded as is into `data`, and corresponding is_lendist bit set to 0
  - second symbol is a length=6 code: subtract 128 from it and store in `data` as 132, and set corresponding bit in is_lendist to 1
    - this length symbol doesn't have any extra bits, so just proceed
  - third symbol is a distance={5 or 6} code: add 160 to it and store in `data` as 164, and set corresponding bit in is_lendist to 1
    - this distance symbol does have 1 extra bit, of value 0, so store 1 in `data`, set is_lendist bit to 1, _and_ store the extra bit value of 0 into xbits
  - the next symbol is a length=258 symbol; subtract 128 and store in `data` as 157, with is_lendist bit as 1
  - the next symbol is a distance={3073-4096} code: add 160 and store in `data` as 183, with is_lendist bit=1
    - this distance symbol has 10 extra bits, which is split over two bytes as 7+3: so set the next two bytes in `data` to 7 and 3, with corresponding is_lendist bits both as 1
    - the bottom 7 extra bits are stored in xbits as 8, then the top 3 extra bits are stored in xbits as 4 [520 = (4<<7) + 8]
  
  Compared to the normal layout of using 16-bit entries, this memory layout uses much less (cache) memory, and is roughly just as easy to process
 */
struct Lz77Data {
	size_t len;
	// add padding onto the end of these, to not have to worry about masking when reading the data
	alignas(8)  uint64_t is_lendist[(OUTPUT_BUFFER_SIZE+63)/64 + 1];
	alignas(64) uint8_t data[OUTPUT_BUFFER_SIZE + sizeof(__m512i)];
	alignas(64) uint8_t xbits[(OUTPUT_BUFFER_SIZE * 3 + 4) / 5 + sizeof(__m512i)]; // compact buffer; largest size is 3/5 of buf (expected size is much smaller)
	
	inline Lz77Data() {
		// is_lendist is written to using BitWriter, so needs to be zeroed up front
		memset(is_lendist, 0, sizeof(is_lendist));
		len = 0;
	}
};

#endif
