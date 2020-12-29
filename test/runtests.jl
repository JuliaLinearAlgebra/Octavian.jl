import Octavian

import BenchmarkTools
import InteractiveUtils
import LinearAlgebra
import Test

using Test: @testset, @test, @test_throws

include("test_suite_preamble.jl")

include("block_sizes.jl")
include("macrokernels.jl")
include("macros.jl")
include("matmul_coverage.jl")
include("pointer_matrix.jl")
include("utils.jl")

if !coverage
    include("matmul.jl")
end
