module Octavian

using VectorizationBase, ArrayInterface, LoopVectorization

using VectorizationBase: align, AbstractStridedPointer, zstridedpointer,
    static_sizeof, lazymul, StridedPointer, gesp, pause, pick_vector_width_val,
    snum_cache_levels, scache_size, snum_cores, num_cores, cache_inclusivity, scacheline_size
using LoopVectorization: maybestaticsize, matmul_params, preserve_buffer, CloseOpen
using ArrayInterface: StaticInt, Zero, One, OptionallyStaticUnitRange, size, strides, offsets, indices,
    static_length, static_first, static_last, axes, dense_dims, stride_rank

using ThreadingUtilities:
    _atomic_add!, _atomic_umax!, _atomic_umin!,
    _atomic_load, _atomic_store!, _atomic_cas_cmp!,
    SPIN, WAIT, TASK, LOCK, STUP, taskpointer,
    wake_thread!, __wait, load, store!

export StaticInt
export matmul!
export matmul
export matmul_serial!
export matmul_serial

include("global_constants.jl")
include("types.jl")
include("staticfloats.jl")
include("integerdivision.jl")
include("memory_buffer.jl")
include("block_sizes.jl")
include("funcptrs.jl")
include("macrokernels.jl")
include("utils.jl")
include("matmul.jl")

include("init.jl") # `Octavian.__init__()` is defined in this file

end # module Octavian
