import Octavian

import BenchmarkTools
import InteractiveUtils
import LinearAlgebra
import LoopVectorization
import Random
import Test
import VectorizationBase

using Test: @testset, @test, @test_throws

@info("Sys.CPU_THREADS is $(Sys.CPU_THREADS)")
@info("VectorizationBase.NUM_CORES is $(VectorizationBase.NUM_CORES)")

include("test_suite_preamble.jl")

@info("Running Octavian tests with $(Octavian.OCTAVIAN_NUM_TASKS[]) tasks")

Random.seed!(123)

using Aqua: test_all
test_all(Octavian)

include("utils.jl")
include("block_sizes.jl")
include("init.jl")
include("macrokernels.jl")
include("matmul_coverage.jl")

if !coverage
    include("matmul.jl")
end
