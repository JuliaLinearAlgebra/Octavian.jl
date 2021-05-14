import Octavian

import Aqua
import BenchmarkTools
import InteractiveUtils
import LinearAlgebra
import LoopVectorization
import Random
import Test
import VectorizationBase

using Test: @testset, @test, @test_throws, @inferred

include("test_suite_preamble.jl")

@info("VectorizationBase.num_cores() is $(VectorizationBase.num_cores())")
@info("Octavian.OCTAVIAN_NUM_TASKS[] is $(Octavian.OCTAVIAN_NUM_TASKS[]) tasks")

Random.seed!(123)

include("block_sizes.jl")
include("init.jl")
include("integer_division.jl")
include("macrokernels.jl")
include("matmul_coverage.jl")
include("utils.jl")

if !coverage
    include("matmul_main.jl")
end

include("aqua.jl") # run the Aqua.jl tests last
