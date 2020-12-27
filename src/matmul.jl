
function matmul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}, _α = one(T), _β = zero(T)) where {T}
    Mc, Kc, Nc = block_sizes(T)

    M, K, N = matmul_sizes(C, A, B)

    # Create L2-buffer for `A`; it should be stack-allocated
    Amem = L2Buffer(T)
    Aptr = Base.unsafe_convert(Ptr{T}, Amem);
    Bptr = Base.unsafe_convert(Ptr{T}, BCACHE);
    
    GC.@preserve Amem begin
        α = T(_α); 
        for n ∈ StaticInt{1}():Nc:N
            nsize = min(Int(n + Nc), Int(N + 1)) - n
            β = T(_β)
            for k ∈ StaticInt{1}():Kc:K
                ksize = min(Int(k + Kc), Int(K + 1)) - k
                Bblock = PointerMatrix(Bptr, (ksize,nsize))
                unsafe_copyto_avx!(Bblock, view(B, k:k+ksize-1, n:n+nsize-1))
                for m ∈ StaticInt{1}():Mc:M
                    msize = min(Int(m + Mc), Int(M + 1)) - m
                    Ablock = PointerMatrix(Aptr, (msize, ksize), true)
                    unsafe_copyto_avx!(Ablock, view(A, m:m+msize-1, k:k+ksize-1))
                    
                    Cblock = view(C, m:m+msize-1, n:n+nsize-1)
                    macrokernel!(Cblock, Ablock, Bblock, α, β)
                end
                β = one(T) # re-writing to the same blocks of `C`, so replace original factor with `1`
            end
        end
    end # GC.@preserve
    C
end

function matmul(A::AbstractMatrix{Ta}, B::AbstractMatrix{Tb}) where {Ta, Tb}
    # TODO: use `similar` / make it more generic; ideally should work with `StaticArrays.MArray`
    C = Matrix{promote_type(Ta, Tb)}(undef, size(A,1), size(B,2))
    matmul!(C, A, B)
end


