@inline Base.unsafe_convert(::Type{Ptr{T}}, d::MemoryBuffer) where {T} = Base.unsafe_convert(Ptr{T}, Base.pointer_from_objref(d))
@inline MemoryBuffer{T}(::UndefInitializer, ::StaticInt{L}) where {L,T} = MemoryBuffer{L,T}(undef)
Base.size(::MemoryBuffer{L}) where L = (L,)
@inline Base.similar(::MemoryBuffer{L,T}) where {L,T} = MemoryBuffer{L,T}(undef)
# Base.IndexStyle(::Type{<:MemoryBuffer}) = Base.IndexLinear()
@inline function Base.getindex(m::MemoryBuffer{L,T}, i::Int) where {L,T}
    @boundscheck checkbounds(m, i)
    GC.@preserve m x = vload(pointer(m), VectorizationBase.lazymul(VectorizationBase.static_sizeof(T), i - one(i)))
    x
end
@inline function Base.setindex!(m::MemoryBuffer{L,T}, x, i::Int) where {L,T}
    @boundscheck checkbounds(m, i)
    GC.@preserve m vstore!(pointer(m), convert(T, x), lazymul(static_sizeof(T), i - one(i)))
end
function core_cache_size(::Type{T}, ::Val{N}) where {T,N}
    CS = VectorizationBase.CACHE_SIZE[N]
    if CS === nothing
        nothing
    else
        StaticInt{CS}() ÷ static_sizeof(T)
    end
end
function cache_size(::Type{T}, ::Val{N}) where {T,N}
    CS = VectorizationBase.CACHE_SIZE[N]
    if CS === nothing
        nothing
    else
        CC = StaticInt{VectorizationBase.CACHE_COUNT[N]}()
        (StaticInt{CS}() * CC) ÷ (static_sizeof(T))
    end
end
@inline function cache_buffer(::Type{T}, ::Val{N}) where {T,N}
    CS = VectorizationBase.CACHE_SIZE[N]
    if CS === nothing
        nothing
    else
        MemoryBuffer{T}(undef, StaticInt{CS}() ÷ static_sizeof(T))# + (StaticInt{4096}() ÷ static_sizeof(T)))
    end
end
@inline function core_cache_buffer(::Type{T}, ::Val{N}) where {T,N}
    L = core_cache_size(T, Val{N}())
    L === nothing && return nothing
    MemoryBuffer{T}(undef, L)
end


BCache(i::Integer) = BCache(pointer(BCACHE)+cld_fast(SECOND_CACHE_SIZE*i,Threads.nthreads()), i % UInt)
BCache(::Nothing) = BCache(pointer(BCACHE), nothing)

@inline Base.pointer(b::BCache) = b.p
@inline Base.unsafe_convert(::Type{Ptr{T}}, b::BCache) where {T} = Base.unsafe_convert(Ptr{T}, b.p)

function _use_bcache()
    while Threads.atomic_cas!(BCACHE_LOCK, zero(UInt), typemax(UInt)) != zero(UInt)
        pause()
    end
    return BCache(nothing)
end
@inline _free_bcache!(b::BCache{Nothing}) = reseet_bcache_lock!()

_use_bcache(::Nothing) = _use_bcache()
function _use_bcache(i)
    f = one(UInt) << i
    while (Threads.atomic_or!(BCACHE_LOCK, f) & f) != zero(UInt)
        pause()
    end
    BCache(i)
end
_free_bcache!(b::BCache{UInt}) = (Threads.atomic_xor!(BCACHE_LOCK, one(UInt) << b.i); nothing)

"""
  reset_bcache_lock!()

Currently not using try/finally in matmul routine, despite locking.
So if it errors for some reason, you may need to manually call `reset_bcache_lock!()`.
"""
@inline reseet_bcache_lock!() = (BCACHE_LOCK[] = zero(UInt); nothing)

