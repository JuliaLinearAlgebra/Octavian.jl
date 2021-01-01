evenly_divide(x, y) = cld(x, cld(x, y))
evenly_divide(x, y, z) = cld(evenly_divide(x, y), z) * z

"""
    matmul!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix, _α = 1, _β = 0)
"""
function matmul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, _α = one(T), _β = zero(T)) where {T}
    _Mc, _Kc, _Nc = block_sizes(T)

    M, K, N = matmul_sizes(C, A, B)

    # Check if maybe it's better not to pack at all.
    if M * K ≤ _Mc * _Kc && A isa DenseArray && C isa StridedArray && B isa StridedArray && #
        (stride(A,2) ≤ 72 || (iszero(stride(A,2) & (VectorizationBase.pick_vector_width(eltype(A))-1)) && iszero(reinterpret(Int,pointer(A)) & 63)))
        macrokernel!(C, A, B, _α, _β)
        return C
    end

    # check if we want to skip packing B
    do_not_pack_B = (B isa DenseArray && (K * N ≤ _Kc * _Nc)) || N ≤ LoopVectorization.nᵣ

    Bptr = Base.unsafe_convert(Ptr{T}, BCACHE);

    Mc = evenly_divide(M, _Mc, VectorizationBase.pick_vector_width_val(T) * StaticInt{LoopVectorization.mᵣ}())
    Kc = evenly_divide(M, _Kc)
    Nc = evenly_divide(M, _Nc, StaticInt{LoopVectorization.nᵣ}())

    α = T(_α);
    for n ∈ StaticInt{1}():Nc:N # loop 5
        nsize = min(Int(n + Nc), Int(N + 1)) - n
        β = T(_β)
        for k ∈ StaticInt{1}():Kc:K # loop 4
            ksize = min(Int(k + Kc), Int(K + 1)) - k
            Bview = view(B, k:k+ksize-1, n:n+nsize-1)
            # seperate out loop 3, because of _Bblock type instability
            if do_not_pack_B
                # _Bblock is likely to have the same type as _Bblock; it'd be nice to reduce the amount of compilation
                # by homogenizing types across branches, but for now I'm prefering the simplicity of using `Bview`
                # _Bblock = PointerMatrix(gesp1(stridedpointer(B), (k,n)), ksize, nsize)
                # matmul_loop3!(C, T, Ablock, A, _Bblock, α, β, msize, ksize, nsize, M, k, n, Mc)
                matmul_loop3!(C, T, A, Bview, α, β, ksize, nsize, M, k, n, Mc)
            else
                Bblock = PointerMatrix(Bptr, (ksize,nsize))
                unsafe_copyto_avx!(Bblock, Bview)
                matmul_loop3!(C, T, A, Bblock, α, β, ksize, nsize, M, k, n, Mc)
            end
            β = one(T) # re-writing to the same blocks of `C`, so replace original factor with `1`
        end # loop 4
    end # loop 5
    C
end

function matmul_loop3!(C, ::Type{T}, A, Bblock, α, β, ksize, nsize, M, k, n, Mc) where {T}
    full_range = StaticInt{1}():Mc:M
    partitions = Iterators.partition(full_range, OCTAVIAN_NUM_TASKS[])
    @_sync for partition ∈ partitions
        @_spawn begin
            # Create L2-buffer for `A`; it should be stack-allocated
            Amem = L2Buffer(T)
            Aptr = Base.unsafe_convert(Ptr{T}, Amem);
            GC.@preserve Amem begin
                for m ∈ partition # loop 3
                    msize = min(Int(m + Mc), Int(M + 1)) - m
                    Ablock = PointerMatrix(Aptr, (msize, ksize), true)
                    unsafe_copyto_avx!(Ablock, view(A, m:m+msize-1, k:k+ksize-1))

                    Cblock = view(C, m:m+msize-1, n:n+nsize-1)
                    macrokernel!(Cblock, Ablock, Bblock, α, β)
                end # loop 3
            end # GC.@preserve
        end
    end
end

"""
    matmul(A::AbstractMatrix, B::AbstractMatrix)

Return the matrix product A*B.
"""
function matmul(A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}) where {Ta, Tb}
    # TODO: use `similar` / make it more generic; ideally should work with `StaticArrays.MArray`
    C = Matrix{promote_type(Ta, Tb)}(undef, size(A,1), size(B,2))
    matmul!(C, A, B)
end
