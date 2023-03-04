@inline _select(::StaticInt{M}, ::StaticInt{M}) where {M} = StaticInt{M}()
@noinline _select(::StaticInt{M}, ::StaticInt{N}) where {M,N} = throw("$M ≠ $N")
@inline _select(::StaticInt{M}, _) where {M} = StaticInt{M}()
@inline _select(_, ::StaticInt{M}) where {M} = StaticInt{M}()
@inline _select(x, _) = x

"""
Checks sizes for compatibility, and preserves the static size information if
given a mix of static and dynamic sizes.
"""
@inline function matmul_sizes(C, A, B)
  MC, NC = static_size(C)
  MA, KA = static_size(A)
  KB, NB = static_size(B)
  @assert ((MC == MA) & (KA == KB) & (NC == NB)) "Size mismatch."
  (_select(MA, MC), _select(KA, KB), _select(NB, NC))
end

unsafe_copyto_turbo!(pB, pA, M, N) = LoopVectorization.@turbo for n ∈
                                                                  CloseOpen(N),
  m ∈ CloseOpen(M)

  pB[m, n] = pA[m, n]
end

function default_stridedpointer_quote(::Type{T}, N, Ot) where {T}
  C = 1
  B = 0
  R = Expr(:tuple)
  o = Expr(:tuple)
  xt = Expr(:tuple)
  st = StaticInt(sizeof(T))
  gf = GlobalRef(Core, :getfield)
  for n ∈ 1:N
    push!(xt.args, Expr(:call, :*, :st, Expr(:call, gf, :x, n, false)))
    push!(R.args, n)
    push!(o.args, Expr(:call, Ot))
  end
  quote
    $(Expr(:meta, :inline))
    st = $st
    si = StrideIndex{$N,$R,$C}($xt, $o)
    stridedpointer(ptr, si, StaticInt{$B}())
  end
end

@generated function default_stridedpointer(
  ptr::Ptr{T},
  x::X
) where {T,N,X<:Tuple{Vararg{Integer,N}}}
  default_stridedpointer_quote(T, N, :One)
end
@generated function default_zerobased_stridedpointer(
  ptr::Ptr{T},
  x::X
) where {T,N,X<:Tuple{Vararg{Integer,N}}}
  default_stridedpointer_quote(T, N, :Zero)
end

@generated function splitfirstdim(
  sp::StridedPointer{T,N,C,B,R},
  ::StaticInt{M}
) where {T,N,C,B,R,M}
  gf = GlobalRef(Core, :getfield)
  fx = M * sizeof(T)
  xt = Expr(:tuple, :x0, Expr(:call, *, :x0, StaticInt(M * sizeof(T))))
  ot = Expr(:tuple, Zero(), Zero())
  R₁ = (R[1])::Int
  Rn = Expr(:tuple, R₁, R₁ + 1)
  for n ∈ 2:N
    push!(xt.args, Expr(:call, gf, :x, n, false))
    push!(ot.args, Zero())
    Rₙ = (R[n])::Int
    push!(Rn.args, Core.ifelse(Rₙ > R₁, Rₙ + 1, Rₙ))
  end
  Cn = Core.ifelse(C > 1, C + 1, C)
  Bn = Core.ifelse(B > 1, B + 1, B)
  quote
    $(Expr(:meta, :inline))
    x = static_strides(sp)
    x0 = $gf(x, 1, false)
    si = StrideIndex{$(N + 1),$Rn,$Cn}($xt, $ot)
    stridedpointer($gf(sp, :p), si, StaticInt{$Bn}())
  end
end
@generated function droplastdim(sp::StridedPointer{T,N,C,B,R}) where {T,N,C,B,R}
  Cn = Core.ifelse(C == N, -1, C)
  Bn = Core.ifelse(B == N, -1, B)
  xt = Expr(:tuple)
  ot = Expr(:tuple)
  rt = Expr(:tuple)
  gf = GlobalRef(Core, :getfield)
  for n ∈ 1:N-1
    push!(xt.args, Expr(:call, gf, :x, n, false))
    push!(ot.args, Expr(:call, gf, :o, n, false))
    push!(rt.args, R[n])
  end
  quote
    $(Expr(:meta, :inline))
    x = static_strides(sp)
    o = offsets(sp)
    si = StrideIndex{$(N - 1),$rt,$Cn}($xt, $ot)
    stridedpointer($gf(sp, :p), si, StaticInt{$Bn}())
  end
end
