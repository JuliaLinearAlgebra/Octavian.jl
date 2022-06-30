
@inline function first_cache_buffer(::Val{T}) where {T}
  first_cache_buffer(Val{T}(), first_cache_size(Val(T)))
end
@inline function first_cache_buffer(::Val{T}, N) where {T}
  reinterpret(Ptr{T}, ACACHEPTR[] + ((Threads.threadid()-1) * N) * static_sizeof(T))
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
@inline _free_bcache!(::BCache{Nothing}) = reset_bcache_lock!()

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
@inline reset_bcache_lock!() = (BCACHE_LOCK[] = zero(UInt); nothing)

