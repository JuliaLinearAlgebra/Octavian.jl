real_rep(a::AbstractArray{Complex{T}, N}) where {T, N} = reinterpret(reshape, T, a)
#PtrArray(Ptr{T}(pointer(a)), (StaticInt(2), size(a)...))

@inline function _matmul!(_C::AbstractMatrix{Complex{T}}, _A::AbstractMatrix{Complex{U}}, _B::AbstractMatrix{Complex{V}},
                         α=One(), β=Zero(), nthread::Nothing=nothing, MKN=nothing, contig_axis=nothing) where {T,U,V}
    C, A, B =  real_rep.((_C, _A, _B))

    η = ifelse(ArrayInterface.is_lazy_conjugate(_A), StaticInt(-1), StaticInt(1))
    θ = ifelse(ArrayInterface.is_lazy_conjugate(_B), StaticInt(-1), StaticInt(1))
    (+ᶻ, -ᶻ) = ifelse(ArrayInterface.is_lazy_conjugate(_C), (-, +), (+, -))
    ηθ = η*θ

    @avxt for n ∈ indices((C, B), 3), m ∈ indices((C, A), 2)
        Cmn_re = zero(T)
        Cmn_im = zero(T)
        for k ∈ indices((A, B), (3, 2))
            Cmn_re +=     A[1, m, k] * B[1, k, n] - ηθ * A[2, m, k] * B[2, k, n]
            Cmn_im += θ * A[1, m, k] * B[2, k, n] +  η * A[2, m, k] * B[1, k, n]
        end
        C[1,m,n] = (real(α) * Cmn_re -ᶻ imag(α) * Cmn_im) + (real(β) * C[1,m,n] -ᶻ imag(β) * C[2,m,n])
        C[2,m,n] = (imag(α) * Cmn_re +ᶻ real(α) * Cmn_im) + (imag(β) * C[1,m,n] +ᶻ real(β) * C[2,m,n])
    end
    _C
end

@inline function _matmul!(_C::AbstractMatrix{Complex{T}}, A::AbstractMatrix{U}, _B::AbstractMatrix{Complex{V}},
                         α=One(), β=Zero(), nthread::Nothing=nothing, MKN=nothing, contig_axis=nothing) where {T,U,V}
    C, B = real_rep.((_C, _B))
    
    θ = ifelse(ArrayInterface.is_lazy_conjugate(_B), StaticInt(-1), StaticInt(1))
    (+ᶻ, -ᶻ) = ifelse(ArrayInterface.is_lazy_conjugate(_C), (-, +), (+, -))

    @avxt for n ∈ indices((C, B), 3), m ∈ indices((C, A), (2, 1))
        Cmn_re = zero(T)
        Cmn_im = zero(T)
        for k ∈ indices((A, B), (2, 2))
            Cmn_re +=     A[m, k] * B[1, k, n]
            Cmn_im += θ * A[m, k] * B[2, k, n]
        end
        C[1,m,n] = (real(α) * Cmn_re -ᶻ imag(α) * Cmn_im) + (real(β) * C[1,m,n] -ᶻ imag(β) * C[2,m,n])
        C[2,m,n] = (imag(α) * Cmn_re +ᶻ real(α) * Cmn_im) + (imag(β) * C[1,m,n] +ᶻ real(β) * C[2,m,n])
    end
    _C
end

@inline function _matmul!(_C::AbstractMatrix{Complex{T}}, _A::AbstractMatrix{Complex{U}}, B::AbstractMatrix{V},
                         α=One(), β=Zero(), nthread::Nothing=nothing, MKN=nothing, contig_axis=nothing) where {T,U,V}
    C, A = real_rep.((_C, _A))

    η = ifelse(ArrayInterface.is_lazy_conjugate(_A), StaticInt(-1), StaticInt(1))
    (+ᶻ, -ᶻ) = ifelse(ArrayInterface.is_lazy_conjugate(_C), (-, +), (+, -))
    
    @avxt for n ∈ indices((C, B), (3, 2)), m ∈ indices((C, A), 2)
        Cmn_re = zero(T)
        Cmn_im = zero(T)
        for k ∈ indices((A, B), (3, 1))
            Cmn_re +=     A[1, m, k] * B[k, n]
            Cmn_im += η * A[2, m, k] * B[k, n]
        end
        C[1,m,n] = (real(α) * Cmn_re -ᶻ imag(α) * Cmn_im) + (real(β) * C[1,m,n] -ᶻ imag(β) * C[2,m,n])
        C[2,m,n] = (imag(α) * Cmn_re +ᶻ real(α) * Cmn_im) + (imag(β) * C[1,m,n] +ᶻ real(β) * C[2,m,n])
    end
    _C
end





@inline function _matmul_serial!(_C::AbstractMatrix{Complex{T}}, _A::AbstractMatrix{Complex{U}}, _B::AbstractMatrix{Complex{V}},
                         α=One(), β=Zero(), MKN=nothing, contig_axis=nothing) where {T,U,V}
    C, A, B = real_rep.((_C, _A, _B))

    η = ifelse(ArrayInterface.is_lazy_conjugate(_A), StaticInt(-1), StaticInt(1))
    θ = ifelse(ArrayInterface.is_lazy_conjugate(_B), StaticInt(-1), StaticInt(1))
    (+ᶻ, -ᶻ) = ifelse(ArrayInterface.is_lazy_conjugate(_C), (-, +), (+, -))
    ηθ = η*θ
    @avxt for n ∈ indices((C, B), 3), m ∈ indices((C, A), 2)
        Cmn_re = zero(T)
        Cmn_im = zero(T)
        for k ∈ indices((A, B), (3, 2))
            Cmn_re +=     A[1, m, k] * B[1, k, n] - ηθ * A[2, m, k] * B[2, k, n]
            Cmn_im += θ * A[1, m, k] * B[2, k, n] +  η * A[2, m, k] * B[1, k, n]
        end
        C[1,m,n] = (real(α) * Cmn_re -ᶻ imag(α) * Cmn_im) + (real(β) * C[1,m,n] -ᶻ imag(β) * C[2,m,n])
        C[2,m,n] = (imag(α) * Cmn_re +ᶻ real(α) * Cmn_im) + (imag(β) * C[1,m,n] +ᶻ real(β) * C[2,m,n])
    end
    _C
end

@inline function _matmul_serial!(_C::AbstractMatrix{Complex{T}}, A::AbstractMatrix{U}, _B::AbstractMatrix{Complex{V}},
                         α=One(), β=Zero(), MKN=nothing, contig_axis=nothing) where {T,U,V}
    C, B = real_rep.((_C, _B))

    θ = ifelse(ArrayInterface.is_lazy_conjugate(_B), StaticInt(-1), StaticInt(1))
    (+ᶻ, -ᶻ) = ifelse(ArrayInterface.is_lazy_conjugate(_C), (-, +), (+, -))
    
    @avx for n ∈ indices((C, B), 3), m ∈ indices((C, A), (2, 1))
        Cmn_re = zero(T)
        Cmn_im = zero(T)
        for k ∈ indices((A, B), (2, 2))
            Cmn_re +=     A[m, k] * B[1, k, n]
            Cmn_im += θ * A[m, k] * B[2, k, n]
        end
        C[1,m,n] = (real(α) * Cmn_re -ᶻ imag(α) * Cmn_im) + (real(β) * C[1,m,n] -ᶻ imag(β) * C[2,m,n])
        C[2,m,n] = (imag(α) * Cmn_re +ᶻ real(α) * Cmn_im) + (imag(β) * C[1,m,n] +ᶻ real(β) * C[2,m,n])
    end
    _C
end

@inline function _matmul_serial!(_C::AbstractMatrix{Complex{T}}, _A::AbstractMatrix{Complex{U}}, B::AbstractMatrix{V},
                         α=One(), β=Zero(), MKN=nothing, contig_axis=nothing) where {T,U,V}
    C, A = real_rep.((_C, _A))

    η = ifelse(ArrayInterface.is_lazy_conjugate(_A), StaticInt(-1), StaticInt(1))
    (+ᶻ, -ᶻ) = ifelse(ArrayInterface.is_lazy_conjugate(_C), (-, +), (+, -))
    
    @avx for n ∈ indices((C, B), (3, 2)), m ∈ indices((C, A), 2)
        Cmn_re = zero(T)
        Cmn_im = zero(T)
        for k ∈ indices((A, B), (3, 1))
            Cmn_re +=     A[1, m, k] * B[k, n]
            Cmn_im += η * A[2, m, k] * B[k, n]
        end
        C[1,m,n] = (real(α) * Cmn_re -ᶻ imag(α) * Cmn_im) + (real(β) * C[1,m,n] -ᶻ imag(β) * C[2,m,n])
        C[2,m,n] = (imag(α) * Cmn_re +ᶻ real(α) * Cmn_im) + (imag(β) * C[1,m,n] +ᶻ real(β) * C[2,m,n])
    end
    _C
end
