
check_sizes(::StaticInt{M}, ::StaticInt{M}) where {M} = StaticInt{M}()
check_sizes(::StaticInt{M}, ::StaticInt{N}) where {M,N} = throw("$M ≠ $N")
check_sizes(::StaticInt{M}, m) where {M} = (@assert M == m; StaticInt{M}())
check_sizes(m, ::StaticInt{M}) where {M} = (@assert M == m; StaticInt{M}())
check_sizes(m, n) = (@assert m == n; m)

"""
Checks sizes for compatibility, and preserves the static size information if
given a mix of static and dynamic sizes.
"""
function matmul_sizes(C, A, B)
    MC, NC = VectorizationBase.ArrayInterface.size(C)
    MA, KA = VectorizationBase.ArrayInterface.size(A)
    KB, NB = VectorizationBase.ArrayInterface.size(B)
    M = check_sizes(MC, MA)
    K = check_sizes(KA, KB)
    N = check_sizes(NC, NB)
    M, K, N
end


function unsafe_copyto_avx!(B, A)
    LoopVectorization.@avx for i ∈ eachindex(B, A)
        B[i] = A[i]
    end
end

