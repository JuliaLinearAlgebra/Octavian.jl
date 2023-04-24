module Octavian

using Requires: @require

using VectorizationBase, StaticArrayInterface, LoopVectorization

using VectorizationBase:
  align,
  AbstractStridedPointer,
  zstridedpointer,
  vsub_nsw,
  assume,
  static_sizeof,
  StridedPointer,
  gesp,
  pause,
  pick_vector_width,
  has_feature
using CPUSummary: cache_size, num_cores, cache_inclusive, cache_linesize
using LoopVectorization: preserve_buffer, CloseOpen, UpperBoundedInteger
using StaticArrayInterface:
  static_size, static_strides, offsets, indices, axes, StrideIndex
const ArrayInterface = StaticArrayInterface
using IfElse: ifelse
using PolyesterWeave
using Static:
  StaticInt,
  Zero,
  One,
  StaticBool,
  True,
  False,
  gt,
  eq,
  StaticFloat64,
  roundtostaticint,
  floortostaticint
using ManualMemory: MemoryBuffer, load, store!

using ThreadingUtilities:
  _atomic_add!, _atomic_load, _atomic_store!, launch, wait, SPIN

using PrecompileTools: @setup_workload, @compile_workload

if !(StaticInt <: Base.Integer)
  const Integer = Union{Base.Integer,StaticInt}
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
  @setup_workload begin
    # Putting some things in `setup` can reduce the size of the
    # precompile file and potentially make loading faster.
    __init__()
    A64 = rand(100, 100)
    A32 = rand(Float32, 100, 100)

    @compile_workload begin
      # All calls in this block will be precompiled, regardless of whether
      # they belong to Octavian.jl or not (on Julia 1.8 and higher).
      matmul(A64, A64)
      matmul(A64', A64)
      matmul(A64, A64')
      matmul(A64', A64')

      matmul(A32, A32)
      matmul(A32', A32)
      matmul(A32, A32')
      matmul(A32', A32')
    end
  end
end

end # module Octavian
