module Octavian

using VectorizationBase, ArrayInterface, LoopVectorization

using VectorizationBase: align, AbstractStridedPointer, zstridedpointer,
    static_sizeof, lazymul, StridedPointer, gesp, pause, pick_vector_width, has_feature,
    num_cache_levels, cache_size, num_cores, num_cores, cache_inclusive, cache_linesize, ifelse
using LoopVectorization: maybestaticsize, matmul_params, preserve_buffer, CloseOpen
using ArrayInterface: OptionallyStaticUnitRange, size, strides, offsets, indices,
    static_length, static_first, static_last, axes, dense_dims, stride_rank
    
using Static: StaticInt, Zero, One, StaticBool, True, False, gt, eq, StaticFloat64,
    roundtostaticint, floortostaticint
using StrideArraysCore: MemoryBuffer

using ThreadingUtilities:
    _atomic_add!, _atomic_load, _atomic_store!,    
    launch, wait, load, store!

export StaticInt
export matmul!
export matmul
export matmul_serial!
export matmul_serial

include("global_constants.jl")
include("types.jl")
include("integerdivision.jl")
include("memory_buffer.jl")
include("block_sizes.jl")
include("funcptrs.jl")
include("macrokernels.jl")
include("utils.jl")
include("matmul.jl")

include("init.jl") # `Octavian.__init__()` is defined in this file

end # module Octavian
