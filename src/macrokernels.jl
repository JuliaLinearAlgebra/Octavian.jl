
@inline incrementp(A::AbstractStridedPointer{<:Base.HWReal,3}, a::Ptr) = VectorizationBase.increment_ptr(A, a, (Zero(), Zero(), One()))
@inline increment2(B::AbstractStridedPointer{<:Base.HWReal,2}, b::Ptr, ::StaticInt{nᵣ}) where {nᵣ} = VectorizationBase.increment_ptr(B, b, (Zero(), StaticInt{nᵣ}()))
@inline increment1(C::AbstractStridedPointer{<:Base.HWReal,2}, c::Ptr, ::StaticInt{mᵣW}) where {mᵣW} = VectorizationBase.increment_ptr(C, c, (StaticInt{mᵣW}(), Zero()))
macro kernel(pack::Bool, ex::Expr)
  ex.head === :for || throw(ArgumentError("Must be a matmul for loop."))
  mincrements = Expr[:(c = increment1(C, c, mᵣW)), :(ãₚ = incrementp(Ãₚ, ãₚ)), :(m = vsub_nsw(m, mᵣW))]
  # massumes = Expr[:(assume(m < mᵣW)),
  #                 :(assume(VectorizationBase.vgt(ãₚ, VectorizationBase.increment_ptr($(esc(:Ãₚ)), ãₚ, (vsub_nsw($(esc(:M)), mᵣW), LoopVectorization.Zero())), $(esc(:Ãₚ))))),
  #                 :(assume(VectorizationBase.vgt(c, VectorizationBase.increment_ptr($(esc(:C)), c, (vsub_nsw($(esc(:M)), mᵣW), LoopVectorization.Zero())), $(esc(:C)))))]
  offsetprecalc = GlobalRef(VectorizationBase,:offsetprecalc)
  preheader = quote
    mᵣ, nᵣ = matmul_params(Val($(esc(:T))))
    mᵣW = pick_vector_width($(esc(:T))) * mᵣ
    m = $(esc(:M)) % Int32
    n = $(esc(:N)) % Int32
    Ãₚ = $(esc(:Ãₚ))
    B = $offsetprecalc($(esc(:B)), Val{(9,9)}())
    C = $offsetprecalc($(esc(:C)), Val{(9,9)}())
    b = pointer(B); c = pointer(C); ãₚ = pointer(Ãₚ)
  end
  if pack
    push!(mincrements, :(a = increment1(A, a, mᵣW)))
    push!(preheader.args, :(A = $(esc(:A))), :(a = pointer(A)))
    areconstruct = Expr[:($(esc(:A)) = VectorizationBase.reconstruct_ptr(A, a))]
    # push!(massumes, :(assume(VectorizationBase.vgt(a, VectorizationBase.increment_ptr($(esc(:A)), a, (vsub_nsw($(esc(:M)), mᵣW), LoopVectorization.Zero())), $(esc(:A))))))
  else
    Ainit = areconstruct = Expr[]
  end
  lvkern = esc(:(@avx inline=true $ex))

  loopnest = quote
    let ãₚ = ãₚ, c = c, $(esc(:B)) = VectorizationBase.reconstruct_ptr(B, b), m = m
      while m ≥ mᵣW#VectorizationBase.vle(a, amax, A)
        let $(esc(:M)) = mᵣW, $(esc(:N)) = nᵣ, $(esc(:Ãₚ)) = VectorizationBase.reconstruct_ptr(droplastdim(Ãₚ), ãₚ), $(esc(:C)) = VectorizationBase.reconstruct_ptr(C, c), $(areconstruct...)
          $lvkern
          $(mincrements...)
        end
      end
      if m > zero(Int32)#vne(a, amax, A)
        let $(esc(:M)) = UpperBoundedInteger((m%UInt)%Int, mᵣW - One()), $(esc(:N)) = nᵣ, $(esc(:Ãₚ)) = VectorizationBase.reconstruct_ptr(droplastdim(Ãₚ), ãₚ), $(esc(:C)) = VectorizationBase.reconstruct_ptr(C, c), $(areconstruct...)
          # $(massumes...)
          $lvkern
        end
      end
    end
  end

  if !pack
    loopnest = quote
      while n ≥ nᵣ#VectorizationBase.vle(c, cmax, C)
        $loopnest
        c = increment2(C, c, nᵣ)
        b = increment2(B, b, nᵣ)
        n = vsub_nsw(n, nᵣ)
      end
      if n > zero(Int32)#vne(c, cmax, C)
        let $(esc(:B)) = VectorizationBase.reconstruct_ptr(B, b), m = m
          while m ≥ mᵣW#VectorizationBase.vle(a, amax, A)
            let $(esc(:M)) = mᵣW, $(esc(:N)) = UpperBoundedInteger((n%UInt)%Int, nᵣ - One()), $(esc(:Ãₚ)) = VectorizationBase.reconstruct_ptr(droplastdim(Ãₚ), ãₚ), $(esc(:C)) = VectorizationBase.reconstruct_ptr(C, c), $(areconstruct...)
              # assume(n < nᵣ)
              # assume((VectorizationBase.vgt(c, VectorizationBase.increment_ptr($(esc(:C)), c, (Zero(), vsub_nsw($(esc(:N)), nᵣ))), $(esc(:C)))))
              # assume((VectorizationBase.vgt(b, VectorizationBase.increment_ptr($(esc(:B)), b, (Zero(), vsub_nsw($(esc(:N)), nᵣ))), $(esc(:B)))))
              $lvkern
              $(mincrements...)
            end
          end
          if m > zero(Int32)#vne(a, amax, A)
            let $(esc(:M)) = UpperBoundedInteger((m%UInt)%Int, mᵣW - One()), $(esc(:N)) = UpperBoundedInteger((n%UInt)%Int, nᵣ - One()), $(esc(:Ãₚ)) = VectorizationBase.reconstruct_ptr(droplastdim(Ãₚ), ãₚ), $(esc(:C)) = VectorizationBase.reconstruct_ptr(C, c), $(areconstruct...)
              # $(massumes...)
              # assume(n < nᵣ)
              # assume((VectorizationBase.vgt(c, VectorizationBase.increment_ptr($(esc(:C)), c, (Zero(), vsub_nsw($(esc(:N)), nᵣ))), $(esc(:C)))))
              # assume((VectorizationBase.vgt(b, VectorizationBase.increment_ptr($(esc(:B)), b, (Zero(), vsub_nsw($(esc(:N)), nᵣ))), $(esc(:B)))))
              $lvkern
            end
          end
        end
      end
    end
  end
  Expr(:block, preheader, loopnest)
end
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
@inline function ploopmul!(C::AbstractStridedPointer{T}, Ãₚ, B, α, β, M, K, N) where {T}
  1+1
  @kernel false for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
    Cₘₙ = zero(eltype(C))
    for k ∈ CloseOpen(K)
      Cₘₙ += Ãₚ[m,k] * B[k,n]
    end
    C[m,n] = α * Cₘₙ + β * C[m,n]
  end
  nothing
end
@inline function packamul!(
    C::AbstractStridedPointer{T}, Ãₚ, A, B,
    α, β, M, K, N
  ) where {T}
  1+1
  @kernel true for n ∈ CloseOpen(N), m ∈ CloseOpen(M)
    Cₘₙ = zero(eltype(C))
    for k ∈ CloseOpen(K)
      Aₘₖ = A[m,k]
      Cₘₙ += Aₘₖ * B[k,n]
      Ãₚ[m,k] = Aₘₖ 
    end
    C[m,n] = α * Cₘₙ + β * C[m,n] 
  end
end
@inline function alloc_a_pack(K, ::Val{T}) where {T}
  buffer = first_cache_buffer(Val(T))
  mᵣ, nᵣ = matmul_params(Val(T))
  mᵣW = mᵣ * pick_vector_width(T)
  bufferptr = if Sys.WORD_SIZE == 32
    buffer
  else
    align(pointer(buffer))
  end
  Apack = default_zerobased_stridedpointer(bufferptr, (One(), mᵣW, mᵣW * K)) # mᵣW x K x cld(M, mᵣW)
  Apack, buffer
end
function packaloopmul!(
    C::AbstractStridedPointer{T},
    A::AbstractStridedPointer,
    B::AbstractStridedPointer,
    α, β, M, K, N
) where {T}
  Ãₚ, buffer = alloc_a_pack(K, Val(T))
  GC.@preserve buffer begin
    Mᵣ, Nᵣ = matmul_params(Val(T))
    packamul!(C, Ãₚ, A, B, α, β, M, K, Nᵣ)
    ploopmul!(gesp(C, (Zero(), Nᵣ)), Ãₚ, gesp(B, (Zero(), Nᵣ)), α, β, M, K, N - Nᵣ)
  end
  nothing
end

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


