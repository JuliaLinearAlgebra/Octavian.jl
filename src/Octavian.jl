module Octavian

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

# TODO: This loads ForwardDiff.jl and HyperDualNumbers.jl
#       unconditionally on Julia v1.6 - v1.8.
#       It could be reconsidered when these older versions are not supported
#       anymore. In this case, these packages should be removed from the
#       dependencies and treated only as weak dependency.
if !isdefined(Base, :get_extension)
  include("../ext/ForwardDiffExt.jl")
  include("../ext/HyperDualNumbersExt.jl")
end


end # module Octavian
