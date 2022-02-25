
real_rep(a::AbstractArray{DualT}) where {TAG, T, DualT<:ForwardDiff.Dual{TAG, T}} = reinterpret(reshape, T, a)

# multiplication of dual vector/matrix by standard matrix from the left
function _matmul!(_C::AbstractVecOrMat{DualT}, A::AbstractMatrix, _B::AbstractVecOrMat{DualT},
                          α=One(), β=Zero(), nthread::Nothing=nothing, MKN=nothing, contig_axis=nothing) where {DualT <: ForwardDiff.Dual}
    B = real_rep(_B)
    C = real_rep(_C)

    @tturbo for n ∈ indices((C, B), 3), m ∈ indices((C, A), (2, 1)), l in indices((C, B), 1)
        Cₗₘₙ = zero(eltype(C))
        for k ∈ indices((A, B), 2)
            Cₗₘₙ += A[m, k] * B[l, k, n]
        end
        C[l, m, n] = α * Cₗₘₙ + β * C[l, m, n]
    end

    _C
end

# multiplication of dual matrix by standard vector/matrix from the right
@inline function _matmul!(_C::AbstractVecOrMat{DualT}, _A::AbstractMatrix{DualT}, B::AbstractVecOrMat,
                          α=One(), β=Zero(), nthread::Nothing=nothing, MKN=nothing) where {TAG, T, DualT <: ForwardDiff.Dual{TAG, T}}
    if all((ArrayInterface.is_dense(_C), ArrayInterface.is_column_major(_C),
            ArrayInterface.is_dense(_A), ArrayInterface.is_column_major(_A)))
        # we can avoid the reshape and call the standard method
        A = reinterpret(T, _A)
        C = reinterpret(T, _C)
        _matmul!(C, A, B, α, β, nthread, nothing)
    else
        # we cannot use the standard method directly
        A = real_rep(_A)
        C = real_rep(_C)

        @tturbo for n ∈ indices((C, B), (3, 2)), m ∈ indices((C, A), 2), l in indices((C, A), 1)
            Cₗₘₙ = zero(eltype(C))
            for k ∈ indices((A, B), (3, 1))
                Cₗₘₙ += A[l, m, k] * B[k, n]
            end
            C[l, m, n] = α * Cₗₘₙ + β * C[l, m, n]
        end
    end

    _C
end

_view1(B::AbstractMatrix) = @view(B[1,:])
_view1(B::AbstractArray{<:Any,3}) = @view(B[1,:,:])
@inline function _matmul!(_C::AbstractVecOrMat{DualT}, _A::AbstractMatrix{DualT}, _B::AbstractVecOrMat{DualT},
                          α=One(), β=Zero(), nthread::Nothing=nothing, MKN=nothing) where {TAG, T, P, DualT <: ForwardDiff.Dual{TAG, T, P}}
  A = real_rep(_A)
  C = real_rep(_C)
  B = real_rep(_B)
  if all((ArrayInterface.is_dense(_C), ArrayInterface.is_column_major(_C),
          ArrayInterface.is_dense(_A), ArrayInterface.is_column_major(_A)))
    # we can avoid the reshape and call the standard method
    Ar = reinterpret(T, _A)
    Cr = reinterpret(T, _C)
    _matmul!(Cr, Ar, _view1(B), α, β, nthread, nothing)
  else
    # we cannot use the standard method directly
    @tturbo for n ∈ indices((C, B), 3), m ∈ indices((C, A), 2), l in indices((C, A), 1)
      Cₗₘₙ = zero(eltype(C))
      for k ∈ indices((A, B), (3, 2))
        Cₗₘₙ += A[l, m, k] * B[1, k, n]
      end
      C[l, m, n] = α * Cₗₘₙ + β * C[l, m, n]
    end
  end
  Pstatic = static(P)
  @tturbo for n ∈ indices((B,C),3), m ∈ indices((A,C),2), p ∈ 1:Pstatic
    Cₚₘₙ = zero(eltype(C))
    for k ∈ indices((A,B),(3,2))
      Cₚₘₙ += A[1,m,k] * B[p+1,k,n]
    end
    C[p+1,m,n] = C[p+1,m,n] + α*Cₚₘₙ
  end
  _C
end
