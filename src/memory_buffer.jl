
if Sys.WORD_SIZE == 32
    @inline function first_cache_buffer(::Type{T}) where {T}
        reinterpret(Ptr{T}, ACACHEPTR[] + Threads.threadid() * first_cache_size() - first_cache_size())
    end
else
    @inline function first_cache_buffer(::Type{T}) where {T}
        MemoryBuffer{T}(undef, first_cache_size(T))
    end
end

BCache(i::Integer) = BCache(BCACHEPTR[]+cld_fast(second_cache_size()*i, Threads.nthreads()), i % UInt)
BCache(::Nothing) = BCache(BCACHEPTR[], nothing)

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

