PointerMatrix(p::P, s::S) where {T,P<:VectorizationBase.AbstractStridedPointer{T},S} = PointerMatrix{T,P,S}(p, s)
Base.size(A::PointerMatrix) = map(Int, A.s)
VectorizationBase.stridedpointer(A::PointerMatrix) = A.p
Base.unsafe_convert(::Type{Ptr{T}}, A::PointerMatrix{T}) where {T} = pointer(A.p)
@inline function Base.getindex(A::PointerMatrix, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    VectorizationBase.vload(VectorizationBase.stridedpointer(A), (i,j))
end
@inline function Base.getindex(A::PointerMatrix, i::Integer)
    @boundscheck checkbounds(A, i)
    VectorizationBase.vload(VectorizationBase.stridedpointer(A), (i-1,))
end
@inline function Base.setindex!(A::PointerMatrix{T}, v, i::Integer, j::Integer) where {T}
    @boundscheck checkbounds(A, i, j)
    VectorizationBase.vstore!(VectorizationBase.stridedpointer(A), convert(T, v), (i,j))
    v
end
@inline function Base.setindex!(A::PointerMatrix{T}, v, i::Integer) where {T}
    @boundscheck checkbounds(A, i)
    VectorizationBase.vstore!(VectorizationBase.stridedpointer(A), convert(T, v), (i-1,))
    v
end

function PointerMatrix(Bptr::Ptr{T}, (M,N), padcols::Bool = false) where {T}
    st = VectorizationBase.static_sizeof(T)
    _M = padcols ? VectorizationBase.align(M, T) : M
    # Should maybe add a more convenient column major constructor
    Bsptr = VectorizationBase.stridedpointer(
        Bptr, VectorizationBase.ArrayInterface.Contiguous{1}(), VectorizationBase.ArrayInterface.ContiguousBatch{0}(),
        VectorizationBase.ArrayInterface.StrideRank{(1,2)}(), (st, _M*st), (StaticInt{1}(),StaticInt{1}())
    )
    PointerMatrix(Bsptr, (M,N))
end
