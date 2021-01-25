
const OCTAVIAN_NUM_TASKS = Ref(1)
_nthreads() = OCTAVIAN_NUM_TASKS[]

@generated function calc_factors(::Union{Val{nc},StaticInt{nc}} = snum_cores()) where {nc}
    t = Expr(:tuple)
    for i ∈ nc:-1:1
        d, r = divrem(nc, i)
        iszero(r) && push!(t.args, (i, d))
    end
    t
end
# const CORE_FACTORS = calc_factors()

@generated function MᵣW_mul_factor()
    f = VectorizationBase.has_feature("x86_64_avx512f") ? 4 : 9
    Expr(:call, Expr(:curly, :StaticInt, f))
end

@generated function W₁Default()
    w = if VectorizationBase.has_feature("x86_64_avx512f")
        0.006089395198610773
    elseif (Sys.CPU_NAME === "znver2") || (Sys.CPU_NAME === "znver3") # these are znver2 values, I'm assuming they're better for znver3 than generic
        0.1
    elseif Sys.CPU_NAME === "znver1"
        0.053918949422353986
    else
        0.1
    end
    Expr(:call, Expr(:curly, :StaticFloat, w))
end
@generated function W₂Default()
    w = if VectorizationBase.has_feature("x86_64_avx512f")
        0.7979822724696168
    elseif (Sys.CPU_NAME === "znver2") || (Sys.CPU_NAME === "znver3") # these are znver2 values, I'm assuming they're better for znver3 than generic
        0.993489411720157
    elseif Sys.CPU_NAME === "znver1"
        0.3013238122374886
    else
        0.15989396641218157
    end
    Expr(:call, Expr(:curly, :StaticFloat, w))
end
@generated function R₁Default()
    w = if VectorizationBase.has_feature("x86_64_avx512f")
        0.5900561503730485
    elseif (Sys.CPU_NAME === "znver2") || (Sys.CPU_NAME === "znver3") # these are znver2 values, I'm assuming they're better for znver3 than generic
        0.6052218809954467
    elseif Sys.CPU_NAME === "znver1"
        0.6077103834481342
    else
        0.4203583148344484
    end
    Expr(:call, Expr(:curly, :StaticFloat, w))
end
@generated function R₂Default()
    w = if VectorizationBase.has_feature("x86_64_avx512f")
        0.762152930709678
    elseif (Sys.CPU_NAME === "znver2") || (Sys.CPU_NAME === "znver3") # these are znver2 values, I'm assuming they're better for znver3 than generic
        0.7594052633561165
    elseif Sys.CPU_NAME === "znver1"
        0.8775382433240162
    else
        0.6344856142604789
    end
    Expr(:call, Expr(:curly, :StaticFloat, w))
end

first_cache() = StaticInt{1}() + (snum_cache_levels() > StaticInt{2}() ? One() : Zero())
second_cache() = StaticInt{2}() + (snum_cache_levels() > StaticInt{2}() ? One() : Zero())

function first_cache_size()
    fcs = scache_size(first_cache())
    if fcs === Zero()
        return StaticInt(262144)
    elseif (first_cache() === StaticInt(2)) && cache_inclusivity()[2]
        return fcs - scache_size(One())
    else
        return fcs
    end
end
function second_cache_size()
    scs = scache_size(second_cache())
    if scs === Zero()
        return StaticInt(3145728)
    elseif cache_inclusivity()[second_cache()]
        return scs - scache_size(first_cache())
    else
        return scs
    end
end
first_cache_size(::Type{T}) where {T} = first_cache_size() ÷ static_sizeof(T)
second_cache_size(::Type{T}) where {T} = second_cache_size() ÷ static_sizeof(T)

bcache_count() = VectorizationBase.scache_count(second_cache())

const BCACHEPTR = Ref{Ptr{Cvoid}}(C_NULL)
const BCACHE_LOCK = Threads.Atomic{UInt}(zero(UInt))

if Sys.WORD_SIZE ≤ 32
    const ACACHEPTR = Ref{Ptr{Cvoid}}(C_NULL)
end

