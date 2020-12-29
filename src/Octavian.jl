module Octavian

import LoopVectorization
import VectorizationBase

using VectorizationBase: StaticInt

export matmul
export matmul!

include("macros.jl")
include("types.jl")

include("block_sizes.jl")
include("macrokernels.jl")
include("matmul.jl")
include("memory_buffer.jl")
include("pointer_matrix.jl")
include("utils.jl")

const BCACHE = UInt8[]

function __init__()
    resize!(BCACHE, VectorizationBase.CACHE_SIZE[3] * VectorizationBase.CACHE_COUNT[3]);
end

end # module Octavian
