# Octavian

[![Documentation (stable)][docs-stable-img]][docs-stable-url]
[![Documentation (dev)][docs-dev-img]][docs-dev-url]
[![Continuous Integration][ci-img]][ci-url]
[![Continuous Integration (Julia nightly)][ci-julia-nightly-img]][ci-julia-nightly-url]
[![Code Coverage][codecov-img]][codecov-url]

[docs-stable-url]:      https://JuliaLinearAlgebra.org/Octavian.jl/stable
[docs-dev-url]:         https://JuliaLinearAlgebra.org/Octavian.jl/dev
[ci-url]:               https://github.com/JuliaLinearAlgebra/Octavian.jl/actions?query=workflow%3ACI
[ci-julia-nightly-url]: https://github.com/JuliaLinearAlgebra/Octavian.jl/actions?query=workflow%3A%22CI+%28Julia+nightly%29%22
[codecov-url]:          https://codecov.io/gh/JuliaLinearAlgebra/Octavian.jl

[docs-stable-img]:      https://img.shields.io/badge/docs-stable-blue.svg                                            "Documentation (stable)"
[docs-dev-img]:         https://img.shields.io/badge/docs-dev-blue.svg                                               "Documentation (dev)"
[ci-img]:               https://github.com/JuliaLinearAlgebra/Octavian.jl/workflows/CI/badge.svg                     "Continuous Integration"
[ci-julia-nightly-img]: https://github.com/JuliaLinearAlgebra/Octavian.jl/workflows/CI%20(Julia%20nightly)/badge.svg "Continuous Integration (Julia nightly)"
[codecov-img]:          https://codecov.io/gh/JuliaLinearAlgebra/Octavian.jl/branch/master/graph/badge.svg           "Code Coverage"

Octavian.jl
is a multi-threaded BLAS-like library that provides pure Julia
matrix multiplication on the CPU, built on top of
[LoopVectorization.jl](https://github.com/chriselrod/LoopVectorization.jl).

Please see the
[Octavian documentation](https://JuliaLinearAlgebra.org/Octavian.jl/stable).

## Benchmarks

You can run benchmarks using [BLASBenchmarksCPU.jl](https://github.com/JuliaLinearAlgebra/BLASBenchmarksCPU.jl):
```julia
julia> @time using BLASBenchmarksCPU
  7.278954 seconds (17.59 M allocations: 1.107 GiB, 6.22% gc time)

julia> rb = runbench(sizes = logspace(10, 1_000, 200)); plot(rb, displayplot = false);
Progress: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 2:25:04
  Size:               (1000, 1000, 1000)
  BLIS:               (MedianGFLOPS = 1051.0, MaxGFLOPS = 1476.0)
  Gaius:              (MedianGFLOPS = 765.8, MaxGFLOPS = 941.7)
  MKL:                (MedianGFLOPS = 1348.0, MaxGFLOPS = 1589.0)
  Octavian:           (MedianGFLOPS = 1816.0, MaxGFLOPS = 1895.0)
  OpenBLAS:           (MedianGFLOPS = 1254.0, MaxGFLOPS = 1385.0)
  Tullio:             (MedianGFLOPS = 1102.0, MaxGFLOPS = 1196.0)
  LoopVectorization:  (MedianGFLOPS = 1552.0, MaxGFLOPS = 1721.0)

julia> versioninfo()
Julia Version 1.7.0-DEV.1124
Commit d18cf93bac* (2021-05-19 16:11 UTC)
Platform Info:
  OS: Linux (x86_64-generic-linux)
  CPU: Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, cascadelake)
Environment:
  JULIA_NUM_THREADS = 36
```
Resulted in the following:
![octavian10980xebench](https://raw.githubusercontent.com/JuliaLinearAlgebra/Octavian.jl/master/docs/src/assets/bench10980xe.svg)

## Related Packages

| Julia Package                                                    | CPU | GPU |
| ---------------------------------------------------------------- | --- | --- |
| [Gaius.jl](https://github.com/MasonProtter/Gaius.jl)             | Yes | No  |
| [GemmKernels.jl](https://github.com/JuliaGPU/GemmKernels.jl)     | No  | Yes |
| [Octavian.jl](https://github.com/JuliaLinearAlgebra/Octavian.jl) | Yes | No  |
| [Tullio.jl](https://github.com/mcabbott/Tullio.jl)               | Yes | Yes |

In general:
- Octavian has the fastest CPU performance.
- GemmKernels has the fastest GPU performance.
- Tullio is the most flexible.
