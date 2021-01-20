const BCACHE = UInt8[]

const OCTAVIAN_NUM_TASKS = Ref(1)
_nthreads() = OCTAVIAN_NUM_TASKS[]

@generated function calc_factors(::Val{nc} = Val{NUM_CORES}()) where {nc}
    t = Expr(:tuple)
    for i ∈ nc:-1:1
        d, r = divrem(nc, i)
        iszero(r) && push!(t.args, (i, d))
    end
    t
end
const CORE_FACTORS = calc_factors()

const MᵣW_mul_factor = VectorizationBase.REGISTER_SIZE === 64 ? StaticInt{4}() : StaticInt{9}()

if VectorizationBase.AVX512F
    const W₁Default = 0.006089395198610773
    const W₂Default = 0.7979822724696168
    const R₁Default = 0.5900561503730485
    const R₂Default = 0.762152930709678
else
    const W₁Default = 0.1 # TODO: relax bounds; this was the upper bound set for the optimizer.
    const W₂Default = 0.15989396641218157
    const R₁Default = 0.4203583148344484
    const R₂Default = 0.6344856142604789
end

const FIRST__CACHE = 1 + (VectorizationBase.CACHE_SIZE[3] !== nothing)
const SECOND_CACHE = 2 + (VectorizationBase.CACHE_SIZE[3] !== nothing)
const FIRST__CACHE_SIZE = VectorizationBase.CACHE_SIZE[FIRST__CACHE] === nothing ? 262144 :
    (((FIRST__CACHE == 2) & CACHE_INCLUSIVITY[2]) ? (VectorizationBase.CACHE_SIZE[2] - VectorizationBase.CACHE_SIZE[1]) :
    VectorizationBase.CACHE_SIZE[FIRST__CACHE])
const SECOND_CACHE_SIZE = (VectorizationBase.CACHE_SIZE[SECOND_CACHE] === nothing ? 3145728 :
    (CACHE_INCLUSIVITY[SECOND_CACHE] ? (VectorizationBase.CACHE_SIZE[SECOND_CACHE] - VectorizationBase.CACHE_SIZE[FIRST__CACHE]) :
    VectorizationBase.CACHE_SIZE[SECOND_CACHE])) * something(VectorizationBase.CACHE_COUNT[SECOND_CACHE], 1)

const CACHELINESIZE = something(VectorizationBase.L₁CACHE.linesize, 64)
const BCACHE_COUNT = something(VectorizationBase.CACHE_COUNT[3], 1);
const BCACHE_LOCK = Threads.Atomic{UInt}(zero(UInt))



