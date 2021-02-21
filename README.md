# Octavian

[![Documentation (stable)][docs-stable-img]][docs-stable-url]
[![Documentation (dev)][docs-dev-img]][docs-dev-url]
[![Continuous Integration][ci-img]][ci-url]
[![Continuous Integration (Julia nightly)][ci-julia-nightly-img]][ci-julia-nightly-url]
[![Code Coverage][codecov-img]][codecov-url]

[docs-stable-url]:      https://JuliaLinearAlgebra.github.io/Octavian.jl/stable
[docs-dev-url]:         https://JuliaLinearAlgebra.github.io/Octavian.jl/dev
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
[Octavian documentation](https://JuliaLinearAlgebra.github.io/Octavian.jl/stable).

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
