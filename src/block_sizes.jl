
matmul_params(::Val{T}) where {T} = LoopVectorization.matmul_params()

function block_sizes(::Val{T}, _α, _β, R₁, R₂) where {T}
  W = pick_vector_width(T)
  Wfloat = StaticFloat64(W)
  α = _α * Wfloat
  β = _β * Wfloat
  L₁ₑ = StaticFloat64(first_cache_size(Val(T))) * R₁
  L₂ₑ = StaticFloat64(second_cache_size(Val(T))) * R₂
  block_sizes(Val(T), W, α, β, L₁ₑ, L₂ₑ)
end
function block_sizes(::Val{T}, W, α, β, L₁ₑ, L₂ₑ) where {T}
  mᵣ, nᵣ = matmul_params(Val(T))
  MᵣW = mᵣ * W

  Mc = floortostaticint(√(L₁ₑ) * √(L₁ₑ * β + L₂ₑ * α) / √(L₂ₑ) / StaticFloat64(MᵣW)) * MᵣW
  Kc = roundtostaticint(√(L₁ₑ) * √(L₂ₑ) / √(L₁ₑ * β + L₂ₑ * α))
  Nc = floortostaticint(√(L₂ₑ) * √(L₁ₑ * β + L₂ₑ * α) / √(L₁ₑ) / StaticFloat64(nᵣ)) * nᵣ

  Mc, Kc, Nc
end
function block_sizes(::Val{T}) where {T}
  block_sizes(Val(T), W₁Default(), W₂Default(), R₁Default(), R₂Default())
end

"""
    split_m(M, Miters_base, W)

Splits `M` into at most `Miters_base` iterations.
For example, if we wish to divide `517` iterations into roughly 7 blocks using multiples of `8`:

```julia
julia> split_m(517, 7, 8)
(72, 2, 69, 7)
```

This suggests we have base block sizes of size `72`, with two iterations requiring an extra remainder of `8 ( = W)`,
and a final block of `69` to handle the remainder. It also tells us that there are `7` total iterations, as requested.
```julia
julia> 80*2 + 72*(7-2-1) + 69
517
```
This is meant to specify roughly the requested amount of blocks, and return relatively even sizes.

This method is used fairly generally.
"""
@inline function split_m(M, _Mblocks, W)
  Miters = cld_fast(M, W)
  Mblocks = min(_Mblocks, Miters)
  Miter_per_block, Mrem = divrem_fast(Miters, Mblocks)
  Mbsize = Miter_per_block * W
  Mremfinal = M - Mbsize * (Mblocks - 1) - Mrem * W
  Mbsize, Mrem, Mremfinal, Mblocks
end

"""
  solve_block_sizes(::Val{T}, M, K, N, α, β, R₂, R₃)

This function returns iteration/blocking descriptions `Mc`, `Kc`, and `Nc` for use when packing both `A` and `B`.

It tries to roughly minimize the cost
```julia
MKN/(Kc*W) + α * MKN/Mc + β * MKN/Nc
```
subject to constraints
```julia
Mc - M ≤ 0
Kc - K ≤ 0
Nc - N ≤ 0
Mc*Kc - L₁ₑ ≤ 0
Kc*Nc - L₂ₑ ≤ 0
```
That is, our constraints say that our block sizes shouldn't be bigger than the actual dimensions, and also that
our packed `A` (`Mc × Kc`) should fit into the first packing cache (generally, actually the `L₂`, and our packed
`B` (`Kc × Nc`) should fit into the second packing cache (generally the `L₃`).

Our cost model consists of three components:
1. Cost of moving data in and out of registers. This is done `(M/Mᵣ * K/Kc * N/Nᵣ)` times and the cost per is `(Mᵣ/W * Nᵣ)`.
2. Cost of moving strips from `B` pack from the low cache levels to the highest cache levels when multiplying `Aₚ * Bₚ`.
   This is done `(M / Mc * K / Kc * N / Nc)` times, and the cost per is proportional to `(Kc * Nᵣ)`.
   `α` is the proportionality-constant parameter.
3. Cost of packing `A`. This is done `(M / Mc * K / Kc * N / Nc)` times, and the cost per is proportional to
   `(Mc * Kc)`. `β` is the proportionality-constant parameter.

As `W` is a constant, we multiply the cost by `W` and absorb it into `α` and `β`. We drop it from the description
from  here on out.

In the full problem, we would have Lagrangian, with μ < 0:
f((Mc,Kc,Nc),(μ₁,μ₂,μ₃,μ₄,μ₅))
MKN/Kc + α * MKN/Mc + β * MKN/Nc - μ₁(Mc - M) - μ₂(Kc - K) - μ₃(Nc - N) - μ₄(Mc*Kc - L2) - μ₅(Kc*Nc - L3)
```julia
0 = ∂L/∂Mc = - α * MKN / Mc² - μ₁ - μ₄*Kc
0 = ∂L/∂Kc = - MKN / Kc² - μ₂ - μ₄*Mc - μ₅*Nc
0 = ∂L/∂Nc = - β * MKN / Nc² - μ₃ - μ₅*Kc
0 = ∂L/∂μ₁ = M - Mc
0 = ∂L/∂μ₂ = K - Kc
0 = ∂L/∂μ₃ = N - Nc
0 = ∂L/∂μ₄ = L₁ₑ - Mc*Kc
0 = ∂L/∂μ₅ = L₂ₑ - Kc*Nc
```
The first 3 constraints complicate things, because they're trivially solved by setting `M = Mc`, `K = Kc`, and `N = Nc`.
But this will violate the last two constraints in general; normally we will be on the interior of the inequalities,
meaning we'd be dropping those constraints. Doing so, this leaves us with:

First, lets just solve the cost w/o constraints 1-3
```julia
0 = ∂L/∂Mc = - α * MKN / Mc² - μ₄*Kc
0 = ∂L/∂Kc = - MKN / Kc² - μ₄*Mc - μ₅*Nc
0 = ∂L/∂Nc = - β * MKN / Nc² - μ₅*Kc
0 = ∂L/∂μ₄ = L₁ₑ - Mc*Kc
0 = ∂L/∂μ₅ = L₂ₑ - Kc*Nc
```
Solving:
```julia
Mc = √(L₁ₑ)*√(L₁ₑ*β + L₂ₑ*α)/√(L₂ₑ)
Kc = √(L₁ₑ)*√(L₂ₑ)/√(L₁ₑ*β + L₂ₑ*α)
Nc = √(L₂ₑ)*√(L₁ₑ*β + L₂ₑ*α)/√(L₁ₑ)
μ₄ = -K*√(L₂ₑ)*M*N*α/(L₁ₑ^(3/2)*√(L₁ₑ*β + L₂ₑ*α))
μ₅ = -K*√(L₁ₑ)*M*N*β/(L₂ₑ^(3/2)*√(L₁ₑ*β + L₂ₑ*α))
```
These solutions are indepedent of matrix size.
The approach we'll take here is solving for `Nc`, `Kc`, and then finally `Mc` one after the other, incorporating sizes.

Starting with `N`, we check how many iterations would be implied by `Nc`, and then choose the smallest value that would
yield that number of iterations. This also ensures that `Nc ≤ N`.
```julia
Niter = cld(N, √(L₂ₑ)*√(L₁ₑ*β + L₂ₑ*α)/√(L₁ₑ))
Nblock, Nrem = divrem(N, Niter)
Nblock_Nrem = Nblock + (Nrem > 0)
```
We have `Nrem` iterations of size `Nblock_Nrem`, and `Niter - Nrem` iterations of size `Nblock`.

We can now make `Nc = Nblock_Nrem` a constant, and solve the remaining three equations again:
```julia
0 = ∂L/∂Mc = - α * MKN / Mc² - μ₄*Kc
0 = ∂L/∂Kc = - MKN / Kc² - μ₄*Mc - μ₅*Ncm
0 = ∂L/∂μ₄ = L₂ₑ - Mc*Kc
```
yielding
```julia
Mc = √(L₁ₑ)*√(α)
Kc = √(L₁ₑ)/√(α)
μ₄ = -K*M*N*√(α)/L₁ₑ^(3/2)
```
We proceed in the same fashion as for `Nc`, being sure to reapply the `Kc * Nc ≤ L₂ₑ` constraint:
```julia
Kiter = cld(K, min(√(L₁ₑ)/√(α), L₂ₑ/Nc))
Kblock, Krem = divrem(K, Ki)
Kblock_Krem = Kblock + (Krem > 0)
```
This leaves `Mc` partitioning, for which, for which we use the constraint `Mc * Kc ≤ L₁ₑ` to set
the initial number of proposed iterations as `cld(M, L₁ₑ / Kcm)` for calling `split_m`.
```julia
Mbsize, Mrem, Mremfinal, Mblocks = split_m(M, cld(M, L₁ₑ / Kcm), StaticInt{W}())
```

Note that for synchronization on `B`, all threads must have the same values for `Kc` and `Nc`.
`K` and `N` will be equal between threads, but `M` may differ. By calculating `Kc` and `Nc`
independently of `M`, this algorithm guarantees all threads are on the same page.
"""
@inline function solve_block_sizes(::Val{T}, M, K, N, _α, _β, R₂, R₃, Wfactor) where {T}
  W = pick_vector_width(T)
  α = _α * W
  β = _β * W
  L₁ₑ = first_cache_size(Val(T)) * R₂
  L₂ₑ = second_cache_size(Val(T)) * R₃

  # Nc_init = round(Int, √(L₂ₑ)*√(α * L₂ₑ + β * L₁ₑ)/√(L₁ₑ))
  Nc_init⁻¹ = √(L₁ₑ) / (√(L₂ₑ) * √(α * L₂ₑ + β * L₁ₑ))

  Niter = cldapproxi(N, Nc_init⁻¹) # approximate `ceil`
  Nblock, Nrem = divrem_fast(N, Niter)
  Nblock_Nrem = Nblock + One()#(Nrem > 0)

  ((Mblock, Mblock_Mrem, Mremfinal, Mrem, Miter), (Kblock, Kblock_Krem, Krem, Kiter)) =
    solve_McKc(Val(T), M, K, Nblock_Nrem, _α, _β, R₂, R₃, Wfactor)

  (Mblock, Mblock_Mrem, Mremfinal, Mrem, Miter),
  (Kblock, Kblock_Krem, Krem, Kiter),
  promote(Nblock, Nblock_Nrem, Nrem, Niter)
end
# Takes Nc, calcs Mc and Kc
@inline function solve_McKc(::Val{T}, M, K, Nc, _α, _β, R₂, R₃, Wfactor) where {T}
  W = pick_vector_width(T)
  Wfloat = StaticFloat64(W)
  α = _α * Wfloat
  # β = _β * Wfloat
  L₁ₑ = first_cache_size(Val(T)) * R₂
  L₂ₑ = second_cache_size(Val(T)) * R₃

  Kc_init⁻¹ = Base.FastMath.max_fast(√(α / L₁ₑ), Nc * inv(L₂ₑ))
  Kiter = cldapproxi(K, Kc_init⁻¹) # approximate `ceil`
  Kblock, Krem = divrem_fast(K, Kiter)
  Kblock_Krem = Kblock + One()

  Mᵣ = Wfactor * W
  Mc_init = floor(Int, Base.FastMath.div_fast(L₁ₑ / Mᵣ, Float64(Kblock_Krem)))
  Mc_init_base = max(0, Mc_init - 1)
  Kblock_summary = promote(Kblock, Kblock_Krem, Krem, Kiter)
  if (Mc_init_base ≠ 0) # Mc_init > 1
    Mbsize = Mc_init_base * Mᵣ
    Mblocks, Mblocks_rem = divrem_fast(M, Mᵣ)
    Miter, Mrem = divrem_fast(Mblocks, Mc_init_base)
    if Miter == 0
      return (0, 0, Int(M)::Int, 0, 1), Kblock_summary
    elseif Miter > Mrem
      Mblock_Mrem = Mbsize + Mᵣ
      Mremfinal = Mbsize + Mblocks_rem
      # @show Mbsize * (Miter - 1 - Mrem) + Mrem * Mblock_Mrem + Mremfinal
      map(Int, (Mbsize, Mblock_Mrem, Mremfinal, Mrem, Miter)), Kblock_summary
    else
      _Mbsize, _Mrem, _Mremfinal, _Miter = split_m(M, Miter + (Mrem ≠ 0), Mᵣ)
      _Mblock_Mrem = _Mbsize + Mᵣ
      return map(Int, (_Mbsize, _Mblock_Mrem, _Mremfinal, _Mrem, _Miter)), Kblock_summary
    end
  else
    Mbsize0 = Int(Mᵣ)
    Mblock_Mrem0 = Int(Mᵣ)
    Miter0, Mremfinal0 = divrem_fast(M, Mᵣ)
    map(Int, (Mbsize0, Mblock_Mrem0, Mremfinal0, 0, Miter0)), Kblock_summary
  end
end

@inline cldapproxi(n, d⁻¹) = Base.fptosi(
  Int,
  Base.FastMath.add_fast(Base.FastMath.mul_fast(n, d⁻¹), 0.9999999999999432),
) # approximate `ceil`
# @inline divapproxi(n, d⁻¹) = Base.fptosi(Int, Base.FastMath.mul_fast(n, d⁻¹)) # approximate `div`

"""
  find_first_acceptable(M, W)

Finds first combination of `Miter` and `Niter` that doesn't make `M` too small while producing `Miter * Niter = num_cores()`.
This would be awkard if there are computers with prime numbers of cores. I should probably consider that possibility at some point.
"""
@inline function find_first_acceptable(::Val{T}, M, W) where {T}
  Mᵣ, _ = matmul_params(Val(T))
  factors = calc_factors()
  for (miter, niter) ∈ factors
    if miter * (StaticInt(2) * Mᵣ * W) ≤ M + (W + W)
      return miter, niter
    end
  end
  last(factors)
end
"""
  divide_blocks(M, Ntotal, _nspawn, W)

Splits both `M` and `N` into blocks when trying to spawn a large number of threads relative to the size of the matrices.
"""
@inline function divide_blocks(::Val{T}, M, Ntotal, _nspawn, W) where {T}
  _nspawn == num_cores() && return find_first_acceptable(Val(T), M, W)
  mᵣ, _ = matmul_params(Val(T))
  Miter = clamp(div_fast(M, W * mᵣ * MᵣW_mul_factor()), 1, _nspawn)
  nspawn = div_fast(_nspawn, Miter)
  if (nspawn ≤ 1) & (Miter < _nspawn)
    # rebalance Miter
    Miter = cld_fast(_nspawn, cld_fast(_nspawn, Miter))
    nspawn = div_fast(_nspawn, Miter)
  end
  Miter, cld_fast(Ntotal, max(2, cld_fast(Ntotal, nspawn)))
end

