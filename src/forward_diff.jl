
real_rep(a::AbstractArray{DualT}) where {TAG, T, DualT<:ForwardDiff.Dual{TAG, T}} = reinterpret(reshape, T, a)

# multiplication of dual vector/matrix by standard matrix from the left
@inline function _matmul!(_C::AbstractVecOrMat{DualT}, A::AbstractMatrix, _B::AbstractVecOrMat{DualT},
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
                          α=One(), β=Zero(), nthread::Nothing=nothing, MKN=nothing, contig_axis=nothing) where {DualT <: ForwardDiff.Dual}
    A = real_rep(_A)
    C = real_rep(_C)

    @tturbo for n ∈ indices((C, B), (3, 2)), m ∈ indices((C, A), 2), l in indices((C, A), 1)
        Cₗₘₙ = zero(eltype(C))
        for k ∈ indices((A, B), (3, 1))
            Cₗₘₙ += A[l, m, k] * B[k, n]
        end
        C[l, m, n] = α * Cₗₘₙ + β * C[l, m, n]
    end

    _C
end
