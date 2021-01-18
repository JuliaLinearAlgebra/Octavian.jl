@inline _select(::StaticInt{M}, ::StaticInt{M}) where {M} = StaticInt{M}()
@noinline _select(::StaticInt{M}, ::StaticInt{N}) where {M,N} = throw("$M ≠ $N")
@inline _select(::StaticInt{M}, _) where {M} = StaticInt{M}()
@inline _select(_, ::StaticInt{M}) where {M} = StaticInt{M}()
@inline _select(x, _) = x

"""
Checks sizes for compatibility, and preserves the static size information if
given a mix of static and dynamic sizes.
"""
@inline function matmul_sizes(C,A,B)
    MC, NC = size(C)
    MA, KA = size(A)
    KB, NB = size(B)
    @assert ((MC == MA) & (KA == KB) & (NC == NB)) "Size mismatch."
    (_select(MA, MC), _select(KA, KB), _select(NB, NC))
end

function unsafe_copyto_avx!(pB, pA, M, N)
    LoopVectorization.@avx for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
        pB[i] = pA[i]
    end
end

