import Octavian

import BenchmarkTools
import InteractiveUtils
import LinearAlgebra
import LoopVectorization
import Test
import VectorizationBase

using Test: @testset, @test, @test_throws

@info("Sys.CPU_THREADS is $(Sys.CPU_THREADS)")
@info("VectorizationBase.NUM_CORES is $(VectorizationBase.NUM_CORES)")

include("test_suite_preamble.jl")

@info("Running Octavian tests with $(Octavian.OCTAVIAN_NUM_TASKS[]) tasks")

include("block_sizes.jl")
include("init.jl")
include("macrokernels.jl")
include("macros.jl")
include("matmul_coverage.jl")
include("pointer_matrix.jl")
include("utils.jl")

if !coverage
    include("matmul.jl")
end
