Base.unsafe_convert(::Type{Ptr{T}}, m::MemoryBuffer) where {T} = Base.unsafe_convert(Ptr{T}, Base.pointer_from_objref(m))
@inline MemoryBuffer(::StaticInt{L}, ::Type{T}) where {L,T} = MemoryBuffer{L,T}(undef)
@inline function L2Buffer(::Type{T}) where {T}
    MemoryBuffer(StaticInt{VectorizationBase.CACHE_SIZE[2]}() ÷ VectorizationBase.static_sizeof(T), T)
end
