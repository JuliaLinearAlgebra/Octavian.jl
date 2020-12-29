import Octavian

import BenchmarkTools
import InteractiveUtils
import LinearAlgebra
import Test

using Test: @test, @testset

include("test-suite-preamble.jl")

include("macros.jl")

if (run_all_tests) || (coverage)
    include("matmul-coverage.jl")
end

if (run_all_tests) || (!coverage)
    include("matmul.jl")
end
