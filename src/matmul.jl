"""
Only packs `A`. Primitively does column-major packing: it packs blocks of `A` into a column-major temporary.
"""
function matmul_st_only_pack_A!(
    C::AbstractStridedPointer{T}, A::AbstractStridedPointer, B::AbstractStridedPointer,
    α, β, M, K, N, ::StaticFloat64{W₁}, ::StaticFloat64{W₂}, ::StaticFloat64{R₁}, ::StaticFloat64{R₂}
) where {T, W₁, W₂, R₁, R₂}

    mᵣ, nᵣ = matmul_params(Val(T))
    ((Mblock, Mblock_Mrem, Mremfinal, Mrem, Miter), (Kblock, Kblock_Krem, Krem, Kiter)) =
        solve_McKc(Val(T), M, K, N, StaticFloat64{W₁}(), StaticFloat64{W₂}(), StaticFloat64{R₁}(), StaticFloat64{R₂}(), mᵣ)
    for ko ∈ CloseOpen(Kiter)
        ksize = ifelse(ko < Krem, Kblock_Krem, Kblock)
        let A = A, C = C
            for mo in CloseOpen(Miter)
                msize = ifelse((mo+1) == Miter, Mremfinal, ifelse(mo < Mrem, Mblock_Mrem, Mblock))
                # if ko == 0
                #     loopmul!(C, A, B, α, β, msize, ksize, N)
                # else
                #     loopmul!(C, A, B, α, One(), msize, ksize, N)
                # end
                if ko == 0
                    packaloopmul!(C, A, B, α, β, msize, ksize, N)
                else
                    packaloopmul!(C, A, B, α, One(), msize, ksize, N)
                end
                A = gesp(A, (msize, Zero()))
                C = gesp(C, (msize, Zero()))
            end
        end
        A = gesp(A, (Zero(), ksize))
        B = gesp(B, (ksize, Zero()))
    end
    nothing
end

function matmul_st_pack_A_and_B!(
    C::AbstractStridedPointer{T}, A::AbstractStridedPointer, B::AbstractStridedPointer, α, β, M, K, N, W₁, W₂, R₁, R₂, tid
) where {T}
    mᵣ, nᵣ = matmul_params(Val(T))
    # TODO: if this is nested in other threaded code, use only a piece of BCACHE and make R₂ (and thus L₂ₑ) smaller
    (Mblock, Mblock_Mrem, Mremfinal, Mrem, Miter), (Kblock, Kblock_Krem, Krem, Kiter), (Nblock, Nblock_Nrem, Nrem, Niter) =
        solve_block_sizes(Val(T), M, K, N, W₁, W₂, R₁, R₂, mᵣ)

    bcache = _use_bcache(tid)
    L3ptr = Base.unsafe_convert(Ptr{T}, pointer(bcache))
    for n ∈ CloseOpen(Niter)
        nsize = ifelse(n < Nrem, Nblock_Nrem, Nblock)
        let A = A, B = B
            for k ∈ CloseOpen(Kiter)
                ksize = ifelse(k < Krem, Kblock_Krem, Kblock)
                _B = default_zerobased_stridedpointer(L3ptr, (One(),ksize))
                unsafe_copyto_turbo!(_B, B, ksize, nsize)
                let A = A, C = C, B = _B
                    for m in CloseOpen(Miter)
                        msize = ifelse((m+1) == Miter, Mremfinal, ifelse(m < Mrem, Mblock_Mrem, Mblock))
                        if k == 0
                            packaloopmul!(C, A, B, α,     β, msize, ksize, nsize)
                        else
                            packaloopmul!(C, A, B, α, One(), msize, ksize, nsize)
                        end
                        A = gesp(A, (msize, Zero()))
                        C = gesp(C, (msize, Zero()))
                    end
                end
                A = gesp(A, (Zero(), ksize))
                B = gesp(B, (ksize, Zero()))
            end
        end
        B = gesp(B, (Zero(), nsize))
        C = gesp(C, (Zero(), nsize))
    end
    _free_bcache!(bcache)
    nothing
end

@inline contiguousstride1(A) = ArrayInterface.contiguous_axis(A) === One()
@inline contiguousstride1(A::AbstractStridedPointer{T,N,1}) where {T,N} = true
# @inline bytestride(A::AbstractArray, i) = VectorizationBase.bytestrides(A)[i]
@inline bytestride(A::AbstractStridedPointer, i) = strides(A)[i]
@inline firstbytestride(A::AbstractStridedPointer) = bytestride(A, One())

@inline function vectormultiple(bytex, ::Type{Tc}, ::Type{Ta}) where {Tc,Ta}
    Wc = pick_vector_width(Tc) * static_sizeof(Ta) - One()
    iszero(bytex & (VectorizationBase.register_size() - One()))
end
@inline function dontpack(pA::AbstractStridedPointer{Ta}, M, K, ::StaticInt{mc}, ::StaticInt{kc}, ::Type{Tc}) where {mc, kc, Tc, Ta}
    (contiguousstride1(pA) &&
         ((((MᵣW_mul_factor() + StaticInt(5)) * pick_vector_width(Tc)) ≥ M) ||
          (vectormultiple(bytestride(pA, StaticInt{2}()), Tc, Ta) && ((M * K) ≤ (mc * kc)) && iszero(reinterpret(Int, pointer(pA)) & (VectorizationBase.register_size() - One())))))
end


@inline function alloc_matmul_product(A::AbstractArray{TA}, B::AbstractMatrix{TB}) where {TA,TB}
  # TODO: if `M` and `N` are statically sized, shouldn't return a `Matrix`.
  M, KA = size(A)
  KB, N = size(B)
  @assert KA == KB "Size mismatch."
  if M === StaticInt(1)
    Vector{promote_type(TA,TB)}(undef, N)', (M, KA, N)
  else
    Matrix{promote_type(TA,TB)}(undef, M, N), (M, KA, N)
  end
end
@inline function alloc_matmul_product(A::AbstractArray{TA}, B::AbstractVector{TB}) where {TA,TB}
  # TODO: if `M` and `N` are statically sized, shouldn't return a `Matrix`.
  M, KA = size(A)
  KB = length(B)
  @assert KA == KB "Size mismatch."
  Vector{promote_type(TA,TB)}(undef, M), (M, KA, One())
end

@inline function matmul_serial(A::AbstractMatrix, B::AbstractVecOrMat)
  C, (M,K,N) = alloc_matmul_product(A, B)
  matmul_serial!(C, A, B, One(), Zero(), (M,K,N), ArrayInterface.contiguous_axis(C))
  return C
end


# These methods must be compile time constant
maybeinline(::Any, ::Any, ::Any, ::Any) = false
function maybeinline(::StaticInt{M}, ::StaticInt{N}, ::Type{T}, ::Val{true}) where {M,N,T}
    mᵣ, nᵣ = matmul_params(Val(T))
    static_sizeof(T) * StaticInt{M}() * StaticInt{N}() < StaticInt{176}() * mᵣ * nᵣ
end
function maybeinline(::StaticInt{M}, ::StaticInt{N}, ::Type{T}, ::Val{false}) where {M,N,T}
    StaticInt{M}() * static_sizeof(T) ≤ StaticInt{2}() * VectorizationBase.register_size()
end


@inline function matmul_serial!(C::AbstractVecOrMat, A::AbstractMatrix, B::AbstractVecOrMat)
    matmul_serial!(C, A, B, One(), Zero(), nothing, ArrayInterface.contiguous_axis(C))
end
@inline function matmul_serial!(C::AbstractVecOrMat, A::AbstractMatrix, B::AbstractVecOrMat, α)
    matmul_serial!(C, A, B, α, Zero(), nothing, ArrayInterface.contiguous_axis(C))
end
@inline function matmul_serial!(C::AbstractVecOrMat, A::AbstractMatrix, B::AbstractVecOrMat, α, β)
    matmul_serial!(C, A, B, α, β, nothing, ArrayInterface.contiguous_axis(C))
end
@inline function matmul_serial!(C::AbstractVecOrMat, A::AbstractMatrix, B::AbstractVecOrMat, α, β, ::Nothing, ::StaticInt{2})
  _matmul_serial!(C', B', A', α, β, nothing)
  return C
end
@inline function matmul_serial!(C::AbstractVecOrMat, A::AbstractMatrix, B::AbstractVecOrMat, α, β, (M,K,N)::Tuple{Vararg{Integer,3}}, ::StaticInt{2})
  _matmul_serial!(C', B', A', α, β, (N,K,M))
  return C
end
@inline function matmul_serial!(C::AbstractVecOrMat, A::AbstractMatrix, B::AbstractVecOrMat, α, β, MKN, ::StaticInt)
    _matmul_serial!(C, A, B, α, β, MKN)
    return C
end

"""
  matmul_serial!(C, A, B[, α = 1, β = 0])

Calculates `C = α * (A * B) + β * C` in place.

A single threaded matrix-matrix-multiply implementation.
Supports dynamically and statically sized arrays.

Organizationally, `matmul_serial!` checks the arrays properties to try and dispatch to an appropriate implementation.
If the arrays are small and statically sized, it will dispatch to an inlined multiply.

Otherwise, based on the array's size, whether they are transposed, and whether the columns are already aligned, it decides to not pack at all, to pack only `A`, or to pack both arrays `A` and `B`.
"""
@inline function _matmul_serial!(
    C::AbstractMatrix{T}, A::AbstractMatrix, B::AbstractMatrix, α, β, MKN
) where {T<:Real}
    M, K, N = MKN === nothing ? matmul_sizes(C, A, B) : MKN
    if M * N == 0
        return
    elseif K == 0
        matmul_only_β!(C, β)
        return
    end
    pA = zstridedpointer(A); pB = zstridedpointer(B); pC = zstridedpointer(C);
    Cb = preserve_buffer(C); Ab = preserve_buffer(A); Bb = preserve_buffer(B);
    Mc, Kc, Nc = block_sizes(Val(T)); mᵣ, nᵣ = matmul_params(Val(T));
    GC.@preserve Cb Ab Bb begin
        if maybeinline(M, N, T, ArrayInterface.is_column_major(A)) # check MUST be compile-time resolvable
            inlineloopmul!(pC, pA, pB, One(), Zero(), M, K, N)
            return
        elseif (nᵣ ≥ N) || dontpack(pA, M, K, Mc, Kc, T)
            loopmul!(pC, pA, pB, α, β, M, K, N)
            return
        else
            matmul_st_pack_dispatcher!(pC, pA, pB, α, β, M, K, N)
            return
        end
    end
end # function

function matmul_only_β!(C::AbstractMatrix{T}, β::StaticInt{0}) where T
    @turbo for i=1:length(C)
        C[i] = zero(T)
    end
end

function matmul_only_β!(C::AbstractMatrix{T}, β) where T
    @turbo for i=1:length(C)
        C[i] = β * C[i]
    end
end

function matmul_st_pack_dispatcher!(pC::AbstractStridedPointer{T}, pA, pB, α, β, M, K, N) where {T}
    Mc, Kc, Nc = block_sizes(Val(T))
    if (contiguousstride1(pB) ? (Kc * Nc ≥ K * N) : (firstbytestride(pB) ≤ 1600))
        matmul_st_only_pack_A!(pC, pA, pB, α, β, M, K, N, W₁Default(), W₂Default(), R₁Default(), R₂Default())
    # elseif notnested !== nothing && notnested
    #     matmul_st_pack_A_and_B!(pC, pA, pB, α, β, M, K, N, W₁Default(), W₂Default(), R₁Default(), R₂Default(), nothing)
    else
        matmul_st_pack_A_and_B!(pC, pA, pB, α, β, M, K, N, W₁Default(), W₂Default(), R₁Default(), R₂Default()/Threads.nthreads(), Threads.threadid() - 1)
    end
    nothing
end


"""
    matmul(A, B)

Multiply matrices `A` and `B`.
"""
@inline function matmul(A::AbstractMatrix, B::AbstractVecOrMat)
  C, (M,K,N) = alloc_matmul_product(A, B)
  matmul!(C, A, B, One(), Zero(), nothing, (M,K,N), ArrayInterface.contiguous_axis(C))
  return C
end

"""
    matmul!(C, A, B[, α, β, max_threads])

Calculates `C = α * A * B + β * C` in place, overwriting the contents of `C`.
It may use up to `max_threads` threads. It will not use threads when nested in other threaded code.
"""
@inline function matmul!(C::AbstractVecOrMat, A::AbstractMatrix, B::AbstractVecOrMat)
    matmul!(C, A, B, One(), Zero(), nothing, nothing, ArrayInterface.contiguous_axis(C))
end
@inline function matmul!(C::AbstractVecOrMat, A::AbstractMatrix, B::AbstractVecOrMat, α)
    matmul!(C, A, B, α, Zero(), nothing, nothing, ArrayInterface.contiguous_axis(C))
end
@inline function matmul!(C::AbstractVecOrMat, A::AbstractMatrix, B::AbstractVecOrMat, α, β)
    matmul!(C, A, B, α, β, nothing, nothing, ArrayInterface.contiguous_axis(C))
end
@inline function matmul!(C::AbstractVecOrMat, A::AbstractMatrix, B::AbstractVecOrMat, α, β, nthread)
    matmul!(C, A, B, α, β, nthread, nothing, ArrayInterface.contiguous_axis(C))
end
@inline function matmul!(C::AbstractVecOrMat, A::AbstractMatrix, B::AbstractVecOrMat, α, β, nthread, ::Nothing, ::StaticInt{2})
  _matmul!(C', B', A', α, β, nthread, nothing)
  return C
end
@inline function matmul!(C::AbstractVecOrMat, A::AbstractMatrix, B::AbstractVecOrMat, α, β, nthread, (M,K,N)::Tuple{Vararg{Integer,3}}, ::StaticInt{2})
  _matmul!(C', B', A', α, β, nthread, (N,K,M))
  return C
end
@inline function matmul!(C::AbstractVecOrMat, A::AbstractMatrix, B::AbstractVecOrMat, α, β, nthread, MKN, ::StaticInt)
  _matmul!(C, A, B, α, β, nthread, MKN)
  return C
end

@inline function dontpack(pA::AbstractStridedPointer{Ta}, M, K, ::StaticInt{mc}, ::StaticInt{kc}, ::Type{Tc}, nspawn) where {mc, kc, Tc, Ta}
    # TODO: perhaps consider K vs kc by themselves?
    (contiguousstride1(pA) && ((M * K) ≤ (mc * kc) * nspawn >>> 1))
end

# passing MKN directly would let osmeone skip the size check.
@inline function _matmul!(C::AbstractMatrix{T}, A, B, α, β, nthread, MKN) where {T<:Real}
    M, K, N = MKN === nothing ? matmul_sizes(C, A, B) : MKN
    if M * N == 0
        return
    elseif K == 0
        matmul_only_β!(C, β)
        return
    end
    W = pick_vector_width(T)
    pA = zstridedpointer(A); pB = zstridedpointer(B); pC = zstridedpointer(C);
    Cb = preserve_buffer(C); Ab = preserve_buffer(A); Bb = preserve_buffer(B);
    mᵣ, nᵣ = matmul_params(Val(T))
    GC.@preserve Cb Ab Bb begin
        if maybeinline(M, N, T, ArrayInterface.is_column_major(A)) # check MUST be compile-time resolvable
            inlineloopmul!(pC, pA, pB, One(), Zero(), M, K, N)
            return
        else
            (nᵣ ≥ N) && @goto LOOPMUL
            if (Sys.ARCH === :x86_64) || (Sys.ARCH === :i686)
                (M*K*N < (StaticInt{4_096}() * W)) && @goto LOOPMUL
            else
                (M*K*N < (StaticInt{32_000}() * W)) && @goto LOOPMUL
            end
            __matmul!(pC, pA, pB, α, β, M, K, N, nthread)
            return
            @label LOOPMUL
            loopmul!(pC, pA, pB, α, β, M, K, N)
            return
        end
    end
end

# This funciton is sort of a `pun`. It splits aggressively (it does a lot of "splitin'"), which often means it will split-N.
function matmulsplitn!(C::AbstractStridedPointer{T}, A, B, α, β, ::StaticInt{Mc}, M, K, N, nspawn, ::Val{PACK}) where {T, Mc, PACK}
    Mᵣ, Nᵣ = matmul_params(Val(T))
    W = pick_vector_width(T)
    MᵣW = Mᵣ*W
    _Mblocks, Nblocks = divide_blocks(Val(T), M, cld_fast(N, Nᵣ), nspawn, W)
    Mbsize, Mrem, Mremfinal, Mblocks = split_m(M, _Mblocks, W)
    # Nblocks = min(N, _Nblocks)
    Nbsize, Nrem = divrem_fast(N, Nblocks)

    _nspawn = Mblocks * Nblocks
    Mbsize_Mrem, Mbsize_ = promote(Mbsize +     W, Mbsize)
    Nbsize_Nrem, Nbsize_ = promote(Nbsize + One(), Nbsize)

    let _A = A, _B = B, _C = C, n = 0, tnum = 0, Nrc = Nblocks - Nrem, Mrc = Mblocks - Mrem, __Mblocks = Mblocks - One()
        while true
            nsize = ifelse(Nblocks > Nrc, Nbsize_Nrem, Nbsize_); Nblocks -= 1
            let _A = _A, _C = _C, __Mblocks = __Mblocks
                while __Mblocks != 0
                    msize = ifelse(__Mblocks ≥ Mrc, Mbsize_Mrem, Mbsize_); __Mblocks -= 1
                    launch_thread_mul!(_C, _A, _B, α, β, msize, K, nsize, (tnum += 1), Val{PACK}())
                    _A = gesp(_A, (msize, Zero()))
                    _C = gesp(_C, (msize, Zero()))
                end
                if Nblocks != 0
                    launch_thread_mul!(_C, _A, _B, α, β, Mremfinal, K, nsize, (tnum += 1), Val{PACK}())
                else
                    call_loopmul!(_C, _A, _B, α, β, Mremfinal, K, nsize, Val{PACK}())
                    waitonmultasks(CloseOpen(One(), _nspawn))
                    return
                end
            end
            _B = gesp(_B, (Zero(), nsize))
            _C = gesp(_C, (Zero(), nsize))
        end
    end
end

function __matmul!(
    C::AbstractStridedPointer{T}, A::AbstractStridedPointer, B::AbstractStridedPointer, α, β, M, K, N, nthread
) where {T}
    Mᵣ, Nᵣ = matmul_params(Val(T))
    W = pick_vector_width(T)
    Mc, Kc, Nc = block_sizes(Val(T))
    MᵣW = Mᵣ*W

    # Not taking the fast path
    # But maybe we don't want to thread anyway
    # Maybe this is nested, or we have ≤ 1 threads
    nt = _nthreads()
    _nthread = nthread === nothing ? nt : min(nt, nthread)
    if _nthread < 2
        matmul_st_pack_dispatcher!(C, A, B, α, β, M, K, N)
        return
    end
    # We are threading, but how many threads?
    nspawn = if (Sys.ARCH === :x86_64) || (Sys.ARCH === :i686)
        clamp(div_fast(M * N, StaticInt{128}() * W), 1, _nthread)
    else
        clamp(div_fast(M * N, StaticInt{256}() * W), 1, _nthread)
    end
    # nkern = cld_fast(M * N,  MᵣW * Nᵣ)
    
    # Approach:
    # Check if we don't want to pack A,
    #    if not, aggressively subdivide
    # if so, check if we don't want to pack B
    #    if not, check if we want to thread `N` loop anyway
    #       if so, divide `M` first, then use ratio of desired divisions / divisions along `M` to calc divisions along `N`
    #       if not, only thread along `M`. These don't need syncing, as we're not packing `B`
    #    if so, `matmul_pack_A_and_B!`
    #
    # MᵣW * (MᵣW_mul_factor - One()) # gives a smaller Mc, then
    # if 2M/nspawn is less than it, we don't don't `A`
    # First check is: do we just want to split aggressively?
    mᵣ, nᵣ = matmul_params(Val(T))
    if dontpack(A, M, K, Mc, Kc, T, nspawn) || (W ≥ M) || (nᵣ*((num_cores() ≥ StaticInt(8)) ? max(nspawn,8) : 8) ≥ N)
        # `nᵣ*nspawn ≥ N` is needed at the moment to avoid accidentally splitting `N` to be `< nᵣ` while packing
        # Should probably handle that with a smarter splitting function...
        matmulsplitn!(C, A, B, α, β, Mc, M, K, N, nspawn, Val{false}())
    elseif (bcache_count() === Zero()) || ((nspawn*(W+W) > M) || (contiguousstride1(B) ? (roundtostaticint(Kc * Nc * R₂Default()) ≥ K * N) : (firstbytestride(B) ≤ 1600)))
        matmulsplitn!(C, A, B, α, β, Mc, M, K, N, nspawn, Val{true}())
    else # TODO: Allow splitting along `N` for `matmul_pack_A_and_B!`
        matmul_pack_A_and_B!(C, A, B, α, β, M, K, N, nspawn, W₁Default(), W₂Default(), R₁Default(), R₂Default())
    end
    nothing
end


# If tasks is [0,1,2,3] (e.g., `CloseOpen(0,4)`), it will wait on `MULTASKS[i]` for `i = [1,2,3]`.
function waitonmultasks(tasks)
    for tid ∈ tasks
        wait(tid)
    end
end

@inline allocref(::StaticInt{N}) where {N} = Ref{NTuple{N,UInt8}}()
function matmul_pack_A_and_B!(
    C::AbstractStridedPointer{T}, A::AbstractStridedPointer, B::AbstractStridedPointer, α, β, M, K, N,
    tospawn::Int, ::StaticFloat64{W₁}, ::StaticFloat64{W₂}, ::StaticFloat64{R₁}, ::StaticFloat64{R₂}#, ::Val{1}
) where {T,W₁,W₂,R₁,R₂}
    W = pick_vector_width(T)
    mᵣ, nᵣ = matmul_params(Val(T))
    mᵣW = mᵣ * W
    # atomicsync = Ref{NTuple{16,UInt}}()
    Mbsize, Mrem, Mremfinal, _to_spawn = split_m(M, tospawn, W) # M is guaranteed to be > W because of `W ≥ M` condition for `jmultsplitn!`...
    atomicsync = allocref((StaticInt{1}()+num_cores())*cache_linesize())
    p = align(reinterpret(Ptr{UInt32}, Base.unsafe_convert(Ptr{UInt8}, atomicsync)))
    GC.@preserve atomicsync begin
        for i ∈ CloseOpen(_to_spawn)
            _atomic_store!(reinterpret(Ptr{UInt64}, p) + i*cache_linesize(), 0x0000000000000000)
        end
        Mblock_Mrem, Mblock_ = promote(Mbsize + W, Mbsize)
        u_to_spawn = _to_spawn % UInt
        tid = 0
        bc = _use_bcache()
        bc_ptr = Base.unsafe_convert(typeof(pointer(C)), pointer(bc))
        last_id = _to_spawn - One()
        for m ∈ CloseOpen(last_id) # ...thus the fact that `CloseOpen()` iterates at least once is okay.
            Mblock = ifelse(m < Mrem, Mblock_Mrem, Mblock_)
            launch_thread_mul!(C, A, B, α, β, Mblock, K, N, p, bc_ptr, m % UInt, u_to_spawn, StaticFloat64{W₁}(),StaticFloat64{W₂}(),StaticFloat64{R₁}(),StaticFloat64{R₂}())
            A = gesp(A, (Mblock, Zero()))
            C = gesp(C, (Mblock, Zero()))
        end
        sync_mul!(C, A, B, α, β, Mremfinal, K, N, p, bc_ptr, last_id % UInt, u_to_spawn, StaticFloat64{W₁}(), StaticFloat64{W₂}(), StaticFloat64{R₁}(), StaticFloat64{R₂}())
        waitonmultasks(CloseOpen(One(), _to_spawn))
    end
    _free_bcache!(bc)
    return
end

function sync_mul!(
    C::AbstractStridedPointer{T}, A::AbstractStridedPointer, B::AbstractStridedPointer, α, β, M, K, N, atomicp::Ptr{UInt32}, bc::Ptr, id::UInt, total_ids::UInt,
    ::StaticFloat64{W₁}, ::StaticFloat64{W₂}, ::StaticFloat64{R₁}, ::StaticFloat64{R₂}
) where {T, W₁, W₂, R₁, R₂}

  (Mblock, Mblock_Mrem, Mremfinal, Mrem, Miter), (Kblock, Kblock_Krem, Krem, Kiter), (Nblock, Nblock_Nrem, Nrem, Niter) =
    solve_block_sizes(Val(T), M, K, N, StaticFloat64{W₁}(), StaticFloat64{W₂}(), StaticFloat64{R₁}(), StaticFloat64{R₂}(), One())

  sync_iters = 0x00000000
  myp = atomicp + id *cache_linesize()
  Npackb_r_div, Npackb_r_rem = divrem_fast(Nblock_Nrem, total_ids)
  Npackb_r_block_rem, Npackb_r_block_ = promote(Npackb_r_div + One(), Npackb_r_div)

  Npackb___div, Npackb___rem = divrem_fast(Nblock, total_ids)
  Npackb___block_rem, Npackb___block_ = promote(Npackb___div + One(), Npackb___div)

  pack_r_offset = Npackb_r_div * id + min(id, Npackb_r_rem)
  pack___offset = Npackb___div * id + min(id, Npackb___rem)

  pack_r_len = ifelse(id < Npackb_r_rem, Npackb_r_block_rem, Npackb_r_block_)
  pack___len = ifelse(id < Npackb___rem, Npackb___block_rem, Npackb___block_)

  for n in CloseOpen(Niter)
    # Krem
    # pack kc x nc block of B
    nfull = n < Nrem
    nsize = ifelse(nfull, Nblock_Nrem, Nblock)
    pack_offset = ifelse(nfull, pack_r_offset, pack___offset)
    pack_len = ifelse(nfull, pack_r_len, pack___len)
    let A = A, B = B
      for k ∈ CloseOpen(Kiter)
        ksize = ifelse(k < Krem, Kblock_Krem, Kblock)
        _B = default_zerobased_stridedpointer(bc, (One(), ksize))
        unsafe_copyto_turbo!(gesp(_B, (Zero(), pack_offset)), gesp(B, (Zero(), pack_offset)), ksize, pack_len)
        # synchronize before starting the multiplication, to ensure `B` is packed
        _mv = _atomic_add!(myp, 0x00000001)
        sync_iters += 0x00000001
        let atomp = atomicp
          for _ ∈ CloseOpen(total_ids)
            while _atomic_load(atomp) ≠ sync_iters
              pause()
            end
            atomp += cache_linesize()
          end
        end
        # multiply
        let A = A, B = _B, C = C
          for m in CloseOpen(Miter)
            msize = ifelse((m+1) == Miter, Mremfinal, ifelse(m < Mrem, Mblock_Mrem, Mblock))
            if k == 0
              packaloopmul!(C, A, B, α,     β, msize, ksize, nsize)
            else
              packaloopmul!(C, A, B, α, One(), msize, ksize, nsize)
            end
            A = gesp(A, (msize, Zero()))
            C = gesp(C, (msize, Zero()))
          end
        end
        _mv = _atomic_add!(myp + 4, 0x00000001)
        A = gesp(A, (Zero(), ksize))
        B = gesp(B, (ksize, Zero()))
        # synchronize on completion so we wait until every thread is done with `Bpacked` before beginning to overwrite it
        let atomp = atomicp
          for _ ∈ CloseOpen(total_ids)
            while _atomic_load(atomp+4) ≠ sync_iters
              pause()
            end
            atomp += cache_linesize()
          end
        end
      end
    end
    B = gesp(B, (Zero(), nsize))
    C = gesp(C, (Zero(), nsize))
  end
  nothing
end

function _matmul!(y::AbstractVector{T}, A::AbstractMatrix, x::AbstractVector, α, β, MKN, contig_axis) where {T<:Real}
  @tturbo for m ∈ indices((A,y),1)
    yₘ = zero(T)
    for n ∈ indices((A,x),(2,1))
      yₘ += A[m,n]*x[n]
    end
    y[m] = α * yₘ + β * y[m]
  end
  return y
end
function _matmul_serial!(y::AbstractVector{T}, A::AbstractMatrix, x::AbstractVector, α, β, MKN) where {T<:Real}
  @turbo for m ∈ indices((A,y),1)
    yₘ = zero(T)
    for n ∈ indices((A,x),(2,1))
      yₘ += A[m,n]*x[n]
    end
    y[m] = α * yₘ + β * y[m]
  end
  return y
end


