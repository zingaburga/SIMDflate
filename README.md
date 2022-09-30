# SIMDflate

This is an experimental speed-oriented [DEFLATE](https://en.wikipedia.org/wiki/Deflate) compression implementation (with zlib/gzip wrapper support) written largely using [AVX-512](https://en.wikipedia.org/wiki/AVX-512) instructions.  It aims to be faster than other DEFLATE compressors at the expense of some drawbacks:

* The implementation is focused on text compression only
* Design restricts achievable compression, meaning it’s only comparable with fastest compression levels on existing implementations
* Current implementation doesn’t support any speed/size tradeoff options
* Requires an x86-64 CPU with **“*Ice Lake* level” AVX-512 support**
* Design is not easily portable to other ISAs

This code serves more as a demonstration of what can be achieved if we disregard compatibility concerns, and perhaps act as a showcase of what can be done with AVX-512.  The limitations means that this isn’t really a general-purpose compressor, but this might be improved.
It's currently very early in development, which means that it isn’t well geared for production use, lacks features/functionality, likely has bugs (not extensively tested), code poorly documented etc.

## Non-goals of this project

* Achieving ‘maximum’ or high compression
* Consistent or reproducible output across all CPUs (i.e. different features/techniques may be enabled/used depending on CPU/arch)
* Platform portability
* Decompression support

## Required AVX-512 support / Compatible CPUs

This implementation makes extensive use of relatively [new AVX-512 instructions introduced in Intel’s Ice Lake](https://branchfree.org/2019/05/29/why-ice-lake-is-important-a-bit-bashers-perspective/) microarchitecture.  In fact, it uses all AVX-512 subsets supported on Ice Lake (as well as the [BMI1 and BMI2 instruction sets](https://en.wikipedia.org/wiki/X86_Bit_manipulation_instruction_set)) except DQ, BITALG and VAES, or put another way, it uses the following [AVX-512 subsets](https://en.wikipedia.org/wiki/AVX-512#New_instructions_by_sets): F, BW, CD, VL, VBMI, VBMI2, GFNI, VPOPCNTDQ, VPCLMULQDQ, IFMA, VNNI.

Don’t worry if the above is confusing - the following is a list of compatible processors at time of writing:

* Intel [Ice Lake](https://ark.intel.com/content/www/us/en/ark/products/codename/74979/products-formerly-ice-lake.html), [Tiger Lake](https://ark.intel.com/content/www/us/en/ark/products/codename/88759/products-formerly-tiger-lake.html) and [Rocket Lake](https://ark.intel.com/content/www/us/en/ark/products/codename/192985/products-formerly-rocket-lake.html)
  * [10th generation Core mobile ‘G’ processors](https://ark.intel.com/content/www/us/en/ark/products/codename/74979/products-formerly-ice-lake.html#@Mobile)
  * 11th generation Core processors
  * Xeon Scalable 3rd generation (non-Cooperlake), including workstation class Xeons
* Intel [desktop Alder Lake](https://ark.intel.com/content/www/us/en/ark/products/codename/147470/products-formerly-alder-lake.html#@Desktop) (12th generation Core) may have [unofficial support](https://github.com/zingaburga/alderlake_avx512/wiki)
  * Expected to be *un*available on [Raptor Lake](https://en.wikipedia.org/wiki/Raptor_Lake) (13th gen Core) and [probably Meteor Lake](https://twitter.com/InstLatX64/status/1552372740409147393) (14th gen Core)
  * Likely will be available on [Alder Lake-X](https://videocardz.com/newz/intel-sapphire-rapids-hedt-fishhawk-falls-cpu-has-been-spotted-with-16-cores) and later Intel *high-end desktop* or *workstation* platforms
* Intel [Sapphire Rapids](https://en.wikipedia.org/wiki/Sapphire_Rapids) (4th gen Xeon Scalable) or later
* AMD [Zen 4](https://en.wikipedia.org/wiki/Zen_4) (most Ryzen 7000 and 4th generation EPYC processors, including [Zen 4c](https://images.anandtech.com/doci/17055/image_2021_11_08T15_17_57_082Z.png) variants) or later
  * As these weren’t available at the time of writing, SIMDflate has been primarily developed/optimised on Intel’s AVX-512 implementation

SIMDflate is *not* compatible with AVX-512 implemented on [Skylake](https://en.wikipedia.org/wiki/Skylake_(microarchitecture)#High-end_desktop_processors_(Skylake-X))/[Cascadelake](https://ark.intel.com/content/www/us/en/ark/products/codename/124664/products-formerly-cascade-lake.html)/[Cooperlake](https://ark.intel.com/content/www/us/en/ark/products/codename/189143/products-formerly-cooper-lake.html), [Cannonlake](https://en.wikipedia.org/wiki/Cannon_Lake_(microprocessor)) or CNS.
