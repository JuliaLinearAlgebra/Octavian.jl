module Octavian

using VectorizationBase, ArrayInterface, LoopVectorization

using VectorizationBase: align, AbstractStridedPointer, zstridedpointer, vsub_nsw, assume,
    static_sizeof, StridedPointer, gesp, pause, pick_vector_width, has_feature,
    cache_size, num_cores, num_cores, cache_inclusive, cache_linesize
using LoopVectorization: preserve_buffer, CloseOpen, UpperBoundedInteger
using ArrayInterface: size, strides, offsets, indices, axes
using IfElse: ifelse
using Polyester
using Static: StaticInt, Zero, One, StaticBool, True, False, gt, eq, StaticFloat64,
    roundtostaticint, floortostaticint
using ManualMemory: MemoryBuffer

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
include("complex_matmul.jl")

include("init.jl") # `Octavian.__init__()` is defined in this file

end # module Octavian
