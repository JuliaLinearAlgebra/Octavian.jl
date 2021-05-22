```@meta
CurrentModule = Octavian
```

# Octavian

[Octavian.jl](https://github.com/JuliaLinearAlgebra/Octavian.jl)
is a multi-threaded BLAS-like library that provides pure Julia
matrix multiplication on the CPU, built on top of
[LoopVectorization.jl](https://github.com/chriselrod/LoopVectorization.jl).

The source code for Octavian is available in the
[GitHub repository](https://github.com/JuliaLinearAlgebra/Octavian.jl).

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

!!! note

    Octavian's tasks can interfere with tasks spawned by Base.Threads, resulting in much slower execution times when used together. This can be avoided by using threading utilities from [Polyester](https://github.com/JuliaSIMD/Polyester.jl) or [LoopVectorization](https://github.com/JuliaSIMD/LoopVectorization.jl/) instead. See this [Discourse post](https://discourse.julialang.org/t/odd-benchmarktools-timings-using-threads-and-octavian/59838) for more information.


