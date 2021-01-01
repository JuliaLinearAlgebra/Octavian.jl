module Octavian

import LoopVectorization
import VectorizationBase

using VectorizationBase: StaticInt

export matmul
export matmul!

include("global_constants.jl")
include("macros.jl")
include("types.jl")

include("block_sizes.jl")
include("macrokernels.jl")
include("matmul.jl")
include("memory_buffer.jl")
include("pointer_matrix.jl")
include("utils.jl")

include("init.jl") # `Octavian.__init__()` is defined in this file

end # module Octavian
