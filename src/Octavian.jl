module Octavian

using Requires: @require

using VectorizationBase, ArrayInterface, LoopVectorization

using VectorizationBase: align, AbstractStridedPointer, zstridedpointer, vsub_nsw, assume,
  static_sizeof, StridedPointer, gesp, pause, pick_vector_width, has_feature
using CPUSummary: cache_size, num_cores, num_threads, cache_inclusive, cache_linesize
using LoopVectorization: preserve_buffer, CloseOpen, UpperBoundedInteger
using ArrayInterface: size, strides, offsets, indices, axes, StrideIndex
using IfElse: ifelse
using PolyesterWeave
using Static: StaticInt, Zero, One, StaticBool, True, False, gt, eq, StaticFloat64,
    roundtostaticint, floortostaticint
using ManualMemory: MemoryBuffer, load, store!

using ThreadingUtilities: _atomic_add!, _atomic_load, _atomic_store!, launch, wait, SPIN

if !(StaticInt <: Base.Integer)
const Integer = Union{Base.Integer, StaticInt}
end

export StaticInt
export matmul!
export matmul
export matmul_serial!
export matmul_serial

debug() = false

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

@static if VERSION >= v"1.8.0-beta1"
  let
    __init__()
    A64 = rand(100,100)
    matmul(A64,A64)
    matmul(A64',A64)
    matmul(A64,A64')
    matmul(A64',A64')
    A32 = rand(Float32,100,100)
    matmul(A32,A32)
    matmul(A32',A32)
    matmul(A32,A32')
    matmul(A32',A32')
  end
end


end # module Octavian
