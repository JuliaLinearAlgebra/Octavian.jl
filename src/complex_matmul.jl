real_rep(a::AbstractArray{Complex{T}, N}) where {T, N} = reinterpret(reshape, T, a)
#PtrArray(Ptr{T}(pointer(a)), (StaticInt(2), size(a)...))

for AT in [:AbstractVector, :AbstractMatrix]  # to avoid ambiguity error
    @eval begin
        function _matmul!(_C::$AT{Complex{T}}, _A::AbstractMatrix{Complex{U}}, _B::$AT{Complex{V}},
                                α=One(), β=Zero(), nthread::Nothing=nothing, MKN=nothing, contig_axis=nothing) where {T,U,V}
            C, A, B = map(real_rep, (_C, _A, _B))

            η = ifelse(ArrayInterface.is_lazy_conjugate(_A), StaticInt(-1), StaticInt(1))
            θ = ifelse(ArrayInterface.is_lazy_conjugate(_B), StaticInt(-1), StaticInt(1))
            (+ᶻ, -ᶻ) = ifelse(ArrayInterface.is_lazy_conjugate(_C), (-, +), (+, -))
            ηθ = η*θ

            @tturbo for n ∈ indices((C, B), 3), m ∈ indices((C, A), 2)
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

        function _matmul_v2!(_C::$AT{Complex{T}}, _A::AbstractMatrix{Complex{U}}, _B::$AT{Complex{V}}, 
                            α=One(), β=Zero(), nthread::Nothing=nothing, MKN=nothing, contig_axis=nothing) where {T,U,V}
            # C, A, B = map(real_rep, (_C, _A, _B))
            C = reinterpret(T, _C)
            A = reinterpret(T, _A)
            B = real_rep(_B)

            η_bool = ifelse(ArrayInterface.is_lazy_conjugate(_A), StaticInt(-1), StaticInt(1))
            θ_bool = ifelse(ArrayInterface.is_lazy_conjugate(_B), StaticInt(-1), StaticInt(1))
            (+ᶻ, -ᶻ) = ifelse(ArrayInterface.is_lazy_conjugate(_C), (-, +), (+, -))
            # ηθ = η*θ

            signs = Vec(ntuple(x -> ifelse(iseven(x), -one(T), one(T)), pick_vector_width(Float64))...)
            if !η_bool & !θ_bool
                cmatmul_ab(C, A, B)

            _C
        end

        function cmatmul_ab!(C, A, B)
            @tturbo vectorize=2 for n ∈ indices((C, B), (2,3)), m ∈ indices((C, A), 1)
                Cmn = zero(T)
                for k ∈ indices((A, B), (2, 2))
                    Amk = A[m,k]
                    Aperm = vpermilps177(Amk)

                    # A B
                    Cmn = vfmaddsub(Amk, B[1,k,n], vfmaddsub(Aperm, B[2,k,n], Cmn))
                    # A^* B
                    # Cmn = signs * vfmsubadd(Amk, B[1,k,n], vfmadd(Aperm, B[2,k,n], Cmn))
                    # A B^*
                    # Cmn = vfmaddsub(Amk, B[1,k,n], vfmaddsub(-Aperm, B[2,k,n], Cmn))
                    # A^* B^*
                    # Cmn = signs * vfmaddsub(Amk, B[1,k,n], vfmsub(Aperm, B[2,k,n], Cmn))

                    # Cmn_re +=     A[1, m, k] * B[1, k, n] - ηθ * A[2, m, k] * B[2, k, n]
                    # Cmn_im += θ * A[1, m, k] * B[2, k, n] +  η * A[2, m, k] * B[1, k, n]
                end
                C[1,m,n] = (real(α) * Cmn_re -ᶻ imag(α) * Cmn_im) + (real(β) * C[1,m,n] -ᶻ imag(β) * C[2,m,n])
                C[2,m,n] = (imag(α) * Cmn_re +ᶻ real(α) * Cmn_im) + (imag(β) * C[1,m,n] +ᶻ real(β) * C[2,m,n])
                C[m, n] = Cmn
            end
        end

        function cmatmul_astarb()

            signs = Vec(ntuple(x -> ifelse(iseven(x), -one(T), one(T)), pick_vector_width(Float64))...)
            
            @tturbo vectorize=2 for n ∈ indices((C, B), (2,3)), m ∈ indices((C, A), 1)
                Cmn = zero(T)
                for k ∈ indices((A, B), (2, 2))
                    Amk = A[m,k]
                    Aperm = vpermilps177(Amk)

                    # A B
                    # Cmn = vfmaddsub(Amk, B[1,k,n], vfmaddsub(Aperm, B[2,k,n], Cmn))
                    # A^* B
                    Cmn = signs * vfmsubadd(Amk, B[1,k,n], vfmadd(Aperm, B[2,k,n], Cmn))
                    # A B^*
                    # Cmn = vfmaddsub(Amk, B[1,k,n], vfmaddsub(-Aperm, B[2,k,n], Cmn))
                    # A^* B^*
                    # Cmn = signs * vfmaddsub(Amk, B[1,k,n], vfmsub(Aperm, B[2,k,n], Cmn))

                    # Cmn_re +=     A[1, m, k] * B[1, k, n] - ηθ * A[2, m, k] * B[2, k, n]
                    # Cmn_im += θ * A[1, m, k] * B[2, k, n] +  η * A[2, m, k] * B[1, k, n]
                end
                C[1,m,n] = (real(α) * Cmn_re -ᶻ imag(α) * Cmn_im) + (real(β) * C[1,m,n] -ᶻ imag(β) * C[2,m,n])
                C[2,m,n] = (imag(α) * Cmn_re +ᶻ real(α) * Cmn_im) + (imag(β) * C[1,m,n] +ᶻ real(β) * C[2,m,n])
                C[m, n] = Cmn
            end
        end

        function cmatmul_abstar()
            @tturbo vectorize=2 for n ∈ indices((C, B), (2,3)), m ∈ indices((C, A), 1)
                Cmn = zero(T)
                for k ∈ indices((A, B), (2, 2))
                    Amk = A[m,k]
                    Aperm = vpermilps177(Amk)

                    # TODO: I don't yet know how to pick the correct branch
                    # based on η and θ.
                    # A B
                    # Cmn = vfmaddsub(Amk, B[1,k,n], vfmaddsub(Aperm, B[2,k,n], Cmn))
                    # A^* B
                    # Cmn = signs * vfmsubadd(Amk, B[1,k,n], vfmadd(Aperm, B[2,k,n], Cmn))
                    # A B^*
                    Cmn = vfmaddsub(Amk, B[1,k,n], vfmaddsub(-Aperm, B[2,k,n], Cmn))
                    # A^* B^*
                    # Cmn = signs * vfmaddsub(Amk, B[1,k,n], vfmsub(Aperm, B[2,k,n], Cmn))

                    # Cmn_re +=     A[1, m, k] * B[1, k, n] - ηθ * A[2, m, k] * B[2, k, n]
                    # Cmn_im += θ * A[1, m, k] * B[2, k, n] +  η * A[2, m, k] * B[1, k, n]
                end
                C[1,m,n] = (real(α) * Cmn_re -ᶻ imag(α) * Cmn_im) + (real(β) * C[1,m,n] -ᶻ imag(β) * C[2,m,n])
                C[2,m,n] = (imag(α) * Cmn_re +ᶻ real(α) * Cmn_im) + (imag(β) * C[1,m,n] +ᶻ real(β) * C[2,m,n])
                C[m, n] = Cmn
            end
        end

        function cmatmul_astarbstar()

            signs = Vec(ntuple(x -> ifelse(iseven(x), -one(T), one(T)), pick_vector_width(Float64))...)

            @tturbo vectorize=2 for n ∈ indices((C, B), (2,3)), m ∈ indices((C, A), 1)
                Cmn = zero(T)
                for k ∈ indices((A, B), (2, 2))
                    Amk = A[m,k]
                    Aperm = vpermilps177(Amk)

                    # A B
                    # Cmn = vfmaddsub(Amk, B[1,k,n], vfmaddsub(Aperm, B[2,k,n], Cmn))
                    # A^* B
                    # Cmn = signs * vfmsubadd(Amk, B[1,k,n], vfmadd(Aperm, B[2,k,n], Cmn))
                    # A B^*
                    # Cmn = vfmaddsub(Amk, B[1,k,n], vfmaddsub(-Aperm, B[2,k,n], Cmn))
                    # A^* B^*
                    Cmn = signs * vfmaddsub(Amk, B[1,k,n], vfmsub(Aperm, B[2,k,n], Cmn))

                    # Cmn_re +=     A[1, m, k] * B[1, k, n] - ηθ * A[2, m, k] * B[2, k, n]
                    # Cmn_im += θ * A[1, m, k] * B[2, k, n] +  η * A[2, m, k] * B[1, k, n]
                end
                C[1,m,n] = (real(α) * Cmn_re -ᶻ imag(α) * Cmn_im) + (real(β) * C[1,m,n] -ᶻ imag(β) * C[2,m,n])
                C[2,m,n] = (imag(α) * Cmn_re +ᶻ real(α) * Cmn_im) + (imag(β) * C[1,m,n] +ᶻ real(β) * C[2,m,n])
                C[m, n] = Cmn
            end
        end

        @inline function _matmul!(_C::$AT{Complex{T}}, A::AbstractMatrix{U}, _B::$AT{Complex{V}},
                                α=One(), β=Zero(), nthread::Nothing=nothing, MKN=nothing, contig_axis=nothing) where {T,U,V}
            C, B = map(real_rep, (_C, _B))
            
            θ = ifelse(ArrayInterface.is_lazy_conjugate(_B), StaticInt(-1), StaticInt(1))
            (+ᶻ, -ᶻ) = ifelse(ArrayInterface.is_lazy_conjugate(_C), (-, +), (+, -))

            @tturbo for n ∈ indices((C, B), 3), m ∈ indices((C, A), (2, 1))
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

        @inline function _matmul!(_C::$AT{Complex{T}}, _A::AbstractMatrix{Complex{U}}, B::$AT{V},
                                α=One(), β=Zero(), nthread::Nothing=nothing, MKN=nothing, contig_axis=nothing) where {T,U,V}
            C, A = map(real_rep, (_C, _A))

            η = ifelse(ArrayInterface.is_lazy_conjugate(_A), StaticInt(-1), StaticInt(1))
            (+ᶻ, -ᶻ) = ifelse(ArrayInterface.is_lazy_conjugate(_C), (-, +), (+, -))
            
            # @tturbo for n ∈ indices((C, B), (3, 2)), m ∈ indices((C, A), 2)
            @turbo for n ∈ indices((C, B), (3, 2)), m ∈ indices((C, A), 2)
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





        @inline function _matmul_serial!(_C::$AT{Complex{T}}, _A::AbstractMatrix{Complex{U}}, _B::$AT{Complex{V}},
                                α=One(), β=Zero(), MKN=nothing, contig_axis=nothing) where {T,U,V}
            C, A, B = map(real_rep, (_C, _A, _B))

            η = ifelse(ArrayInterface.is_lazy_conjugate(_A), StaticInt(-1), StaticInt(1))
            θ = ifelse(ArrayInterface.is_lazy_conjugate(_B), StaticInt(-1), StaticInt(1))
            (+ᶻ, -ᶻ) = ifelse(ArrayInterface.is_lazy_conjugate(_C), (-, +), (+, -))
            ηθ = η*θ
            @turbo for n ∈ indices((C, B), 3), m ∈ indices((C, A), 2)
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

        @inline function _matmul_serial!(_C::$AT{Complex{T}}, A::AbstractMatrix{U}, _B::$AT{Complex{V}},
                                α=One(), β=Zero(), MKN=nothing, contig_axis=nothing) where {T,U,V}
            C, B = map(real_rep, (_C, _B))

            θ = ifelse(ArrayInterface.is_lazy_conjugate(_B), StaticInt(-1), StaticInt(1))
            (+ᶻ, -ᶻ) = ifelse(ArrayInterface.is_lazy_conjugate(_C), (-, +), (+, -))
            
            @turbo for n ∈ indices((C, B), 3), m ∈ indices((C, A), (2, 1))
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

        @inline function _matmul_serial!(_C::$AT{Complex{T}}, _A::AbstractMatrix{Complex{U}}, B::$AT{V},
                                α=One(), β=Zero(), MKN=nothing, contig_axis=nothing) where {T,U,V}
            C, A = map(real_rep, (_C, _A))

            η = ifelse(ArrayInterface.is_lazy_conjugate(_A), StaticInt(-1), StaticInt(1))
            (+ᶻ, -ᶻ) = ifelse(ArrayInterface.is_lazy_conjugate(_C), (-, +), (+, -))
            
            @turbo for n ∈ indices((C, B), (3, 2)), m ∈ indices((C, A), 2)
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
    end
end
