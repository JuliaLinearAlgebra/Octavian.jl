struct PointerMatrix{T,P<:VectorizationBase.AbstractStridedPointer,S<:Tuple{Vararg{Integer,2}}} <: DenseMatrix{T}
    p::P
    s::S
end

mutable struct MemoryBuffer{L,T}
    data::NTuple{L,T}
    MemoryBuffer{L,T}(::UndefInitializer) where {L,T} = new{L,T}()
end
