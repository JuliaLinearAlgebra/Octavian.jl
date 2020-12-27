import Octavian

import BenchmarkTools
import InteractiveUtils
import Test

using InteractiveUtils: versioninfo
using Test: @test, @testset

versioninfo()

@info("Running tests with $(Threads.nthreads()) threads")

include("macros.jl")
include("matmul.jl")
