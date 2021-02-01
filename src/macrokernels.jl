# @inline function loopmul!(C, A, B, ::StaticInt{1}, ::StaticInt{0}, M, K, N)
#     @avx for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Cₘₙ += A[m,k] * B[k,n]
#         end
#         C[m,n] = Cₘₙ
#     end
#     nothing
# end
# @inline function loopmul!(C, A, B, ::StaticInt{1}, ::StaticInt{1}, M, K, N)
#     @avx for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Cₘₙ += A[m,k] * B[k,n]
#         end
#         C[m,n] += Cₘₙ
#     end
#     nothing
# end
# @inline function loopmul!(C, A, B, ::StaticInt{1}, β, M, K, N)
#     @avx for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Cₘₙ += A[m,k] * B[k,n]
#         end
#         C[m,n] = Cₘₙ + β * C[m,n]
#     end
#     nothing
# end
# @inline function loopmul!(C, A, B, ::StaticInt{-1}, ::StaticInt{0}, M, K, N)
#     @avx for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Cₘₙ -= A[m,k] * B[k,n]
#         end
#         C[m,n] = Cₘₙ
#     end
#     nothing
# end
# @inline function loopmul!(C, A, B, ::StaticInt{-1}, ::StaticInt{1}, M, K, N)
#     @avx for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Cₘₙ -= A[m,k] * B[k,n]
#         end
#         C[m,n] += Cₘₙ
#     end
#     nothing
# end
# @inline function loopmul!(C, A, B, ::StaticInt{-1}, β, M, K, N)
#     @avx for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Cₘₙ -= A[m,k] * B[k,n]
#         end
#         C[m,n] = Cₘₙ + β * C[m,n]
#     end
#     nothing
# end
# @inline function loopmul!(C, A, B, α, ::StaticInt{0}, M, K, N)
#     @avx for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Cₘₙ += A[m,k] * B[k,n]
#         end
#         C[m,n] = α * Cₘₙ
#     end
#     nothing
# end

# @inline function loopmul!(C, A, B, α, ::StaticInt{1}, M, K, N)
#     @avx for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Cₘₙ += A[m,k] * B[k,n]
#         end
#         C[m,n] += α * Cₘₙ
#     end
#     nothing
# end
@inline function loopmul!(C, A, B, α, β, M, K, N)
    @avx for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
        Cₘₙ = zero(eltype(C))
        for k ∈ CloseOpen(K)
            Cₘₙ += A[m,k] * B[k,n]
        end
        C[m,n] = α * Cₘₙ + β * C[m,n]
    end
    nothing
end


# @inline function packamul!(
#     C, Ãₚ, A, B,
#     ::StaticInt{1}, ::StaticInt{0},
#     M, K, N
# )
#     @avx for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Aₘₖ = A[m,k]
#             Cₘₙ += Aₘₖ * B[k,n]
#             Ãₚ[m,k] = Aₘₖ 
#         end
#         C[m,n] = Cₘₙ
#     end
# end
# @inline function packamul!(
#     C, Ãₚ, A, B,
#     ::StaticInt{1}, ::StaticInt{1},
#     M, K, N
# )
#     @avx for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Aₘₖ = A[m,k]
#             Cₘₙ += Aₘₖ * B[k,n]
#             Ãₚ[m,k] = Aₘₖ
#         end
#         C[m,n] += Cₘₙ
#     end
# end
# @inline function packamul!(
#     C, Ãₚ, A, B,
#     ::StaticInt{1}, β,
#     M, K, N
# )
#     @avx for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Aₘₖ = A[m,k]
#             Cₘₙ += Aₘₖ * B[k,n]
#             Ãₚ[m,k] = Aₘₖ 
#         end
#         C[m,n] = Cₘₙ + β * C[m,n]
#     end
# end
# @inline function packamul!(
#     C, Ãₚ, A, B,
#     ::StaticInt{-1}, ::StaticInt{0},
#     M, K, N
# )
#     @avx for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Aₘₖ = A[m,k]
#             Cₘₙ -= Aₘₖ * B[k,n]
#             Ãₚ[m,k] = Aₘₖ 
#         end
#         C[m,n] = Cₘₙ
#     end
# end
# @inline function packamul!(
#     C, Ãₚ, A, B,
#     ::StaticInt{-1}, ::StaticInt{1},
#     M, K, N
# )
#     @avx for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Aₘₖ = A[m,k]
#             Cₘₙ -= Aₘₖ * B[k,n]
#             Ãₚ[m,k] = Aₘₖ
#         end
#         C[m,n] += Cₘₙ
#     end
# end
# @inline function packamul!(
#     C, Ãₚ, A, B,
#     ::StaticInt{-1}, β,
#     M, K, N
# )
#     @avx for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Aₘₖ = A[m,k]
#             Cₘₙ -= Aₘₖ * B[k,n]
#             Ãₚ[m,k] = Aₘₖ 
#         end
#         C[m,n] = Cₘₙ + β * C[m,n]
#     end
# end
# @inline function packamul!(
#     C, Ãₚ, A, B,
#     α, ::StaticInt{0},
#     M, K, N
# )
#     @avx for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Aₘₖ = A[m,k]
#             Cₘₙ += Aₘₖ * B[k,n]
#             Ãₚ[m,k] = Aₘₖ 
#         end
#         C[m,n] = α * Cₘₙ
#     end
# end
# @inline function packamul!(
#     C, Ãₚ, A, B,
#     α, ::StaticInt{1},
#     M, K, N
# )
#     @avx for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Aₘₖ = A[m,k]
#             Cₘₙ += Aₘₖ * B[k,n]
#             Ãₚ[m,k] = Aₘₖ 
#         end
#         C[m,n] += α * Cₘₙ
#     end
# end
@inline function packamul!(
    C, Ãₚ, A, B,
    α, β, M, K, N
)
    @avx for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
        Cₘₙ = zero(eltype(C))
        for k ∈ CloseOpen(K)
            Aₘₖ = A[m,k]
            Cₘₙ += Aₘₖ * B[k,n]
            Ãₚ[m,k] = Aₘₖ 
        end
        C[m,n] = α * Cₘₙ + β * C[m,n] 
   end
end
@inline function alloc_a_pack(A, M, ::Type{T}) where {T}
    buffer = first_cache_buffer(T)
    bufferptr = if Sys.WORD_SIZE == 32
        buffer
    else
        align(pointer(buffer))
    end
    Apack = default_zerobased_stridedpointer(bufferptr, (One(), align(M, T)))
    Apack, buffer
end
@inline function packaloopmul!(
    C::AbstractStridedPointer{T},
    A::AbstractStridedPointer,
    B::AbstractStridedPointer,
    α, β, M, K, N
) where {T}
    Ãₚ, buffer = alloc_a_pack(A, M, T)
    GC.@preserve buffer begin
        Mᵣ, Nᵣ = matmul_params()
        packamul!(C, Ãₚ, A, B, α, β, M, K, Nᵣ)
        loopmul!(gesp(C, (Zero(), Nᵣ)), Ãₚ, gesp(B, (Zero(), Nᵣ)), α, β, M, K, N - Nᵣ)
    end
    nothing
end



# @inline function inlineloopmul!(C, A, B, ::StaticInt{1}, ::StaticInt{0}, M, K, N)
#     @avx inline=true for m ∈ CloseOpen(M), n ∈ CloseOpen(N)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Cₘₙ += A[m,k] * B[k,n]
#         end
#         C[m,n] = Cₘₙ
#     end
#     C
# end
# @inline function inlineloopmul!(C, A, B, ::StaticInt{1}, ::StaticInt{1}, M, K, N)
#     @avx inline=true for m ∈ CloseOpen(M), n ∈ CloseOpen(N)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Cₘₙ += A[m,k] * B[k,n]
#         end
#         C[m,n] += Cₘₙ
#     end
#     C
# end
# @inline function inlineloopmul!(C, A, B, ::StaticInt{1}, β, M, K, N)
#     @avx inline=true for m ∈ CloseOpen(M), n ∈ CloseOpen(N)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Cₘₙ += A[m,k] * B[k,n]
#         end
#         C[m,n] = Cₘₙ + β * C[m,n]
#     end
#     C
# end
# @inline function inlineloopmul!(C, A, B, ::StaticInt{-1}, ::StaticInt{0}, M, K, N)
#     @avx inline=true for m ∈ CloseOpen(M), n ∈ CloseOpen(N)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Cₘₙ -= A[m,k] * B[k,n]
#         end
#         C[m,n] = Cₘₙ
#     end
#     C
# end
# @inline function inlineloopmul!(C, A, B, ::StaticInt{-1}, ::StaticInt{1}, M, K, N)
#     @avx inline=true for m ∈ CloseOpen(M), n ∈ CloseOpen(N)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Cₘₙ -= A[m,k] * B[k,n]
#         end
#         C[m,n] += Cₘₙ
#     end
#     C
# end
# @inline function inlineloopmul!(C, A, B, ::StaticInt{-1}, β, M, K, N)
#     @avx inline=true for m ∈ CloseOpen(M), n ∈ CloseOpen(N)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Cₘₙ -= A[m,k] * B[k,n]
#         end
#         C[m,n] = Cₘₙ + β * C[m,n]
#     end
#     C
# end
# @inline function inlineloopmul!(C, A, B, α, ::StaticInt{0}, M, K, N)
#     @avx inline=true for m ∈ CloseOpen(M), n ∈ CloseOpen(N)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Cₘₙ += A[m,k] * B[k,n]
#         end
#         C[m,n] = α * Cₘₙ
#     end
#     C
# end
# @inline function inlineloopmul!(C, A, B, α, ::StaticInt{1}, M, K, N)
#     @avx inline=true for m ∈ CloseOpen(M), n ∈ CloseOpen(N)
#         Cₘₙ = zero(eltype(C))
#         for k ∈ CloseOpen(K)
#             Cₘₙ += A[m,k] * B[k,n]
#         end
#         C[m,n] += α * Cₘₙ
#     end
#     C
# end
@inline function inlineloopmul!(C, A, B, α, β, M, K, N)
    @avx inline=true for m ∈ CloseOpen(M), n ∈ CloseOpen(N)
        Cₘₙ = zero(eltype(C))
        for k ∈ CloseOpen(K)
            Cₘₙ += A[m,k] * B[k,n]
        end
        C[m,n]  = α * Cₘₙ + β * C[m,n]
    end
    C
end


