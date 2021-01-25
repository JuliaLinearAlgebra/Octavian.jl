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
        pB[m,n] = pA[m,n]
    end
end

function default_stridedpointer_quote(::Type{T}, N, Ot) where {T}
    C = 1
    B = 0
    R = Expr(:tuple)
    o = Expr(:tuple)
    xt = Expr(:tuple)
    st = Expr(:call, Expr(:curly, :StaticInt, sizeof(T)))
    for n ∈ 1:N
        push!(xt.args, Expr(:call, :*, :st, Expr(:ref, :x, n)))
        push!(R.args, n)
        push!(o.args, Expr(:call, Ot))
    end
    quote
        $(Expr(:meta,:inline))
        st = $st
        StridedPointer{$T,$N,$C,$B,$R}(ptr, $xt, $o)
    end
end

@generated function default_stridedpointer(ptr::Ptr{T}, x::X) where {T, N, X <: Tuple{Vararg{Integer,N}}}
    default_stridedpointer_quote(T, N, :One)
end
@generated function default_zerobased_stridedpointer(ptr::Ptr{T}, x::X) where {T, N, X <: Tuple{Vararg{Integer,N}}}
    default_stridedpointer_quote(T, N, :Zero)
end


