module Octavian

import LoopVectorization
import VectorizationBase
using VectorizationBase: StaticInt

export matmul!, matmul

include("types.jl")
include("macros.jl")
include("utils.jl")
include("block_sizes.jl")
include("macrokernel.jl")
include("matmul.jl")

const BCACHE = UInt8[]
function __init__()
    resize!(BCACHE, VectorizationBase.CACHE_SIZE[3] * VectorizationBase.CACHE_COUNT[3]);
end


end # module Octavian
