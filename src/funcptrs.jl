

struct LoopMulFunc{P,TC,TA,TB,Α,Β,Md,Kd,Nd} <: Function end
function (::LoopMulFunc{P,TC,TA,TB,Α,Β,Md,Kd,Nd})(p::Ptr{UInt}) where {P,TC,TA,TB,Α,Β,Md,Kd,Nd}
    offset, C = load(p, TC, 1)
    offset, A = load(p, TA, offset)
    offset, B = load(p, TB, offset)
    offset, α = load(p, Α, offset)
    offset, β = load(p, Β, offset)
    offset, M = load(p, Md, offset)
    offset, K = load(p, Kd, offset)
    offset, N = load(p, Nd, offset)
    _call_loopmul!(C, A, B, α, β, M, K, N, Val{P}())
    nothing
end
@inline _call_loopmul!(C, A, B, α, β, M, K, N, ::Val{false}) = loopmul!(C, A, B, α, β, M, K, N)
@inline function _call_loopmul!(C::StridedPointer{T}, A, B, α, β, M, K, N, ::Val{true}) where {T}
    if M*K < first_cache_size(T) * R₂Default()
        packaloopmul!(C, A, B, α, β, M, K, N)
        return
    else
        matmul_st_only_pack_A!(C, A, B, α, β, M, K, N, W₁Default(), W₂Default(), R₁Default(), R₂Default())
        return
    end
end
call_loopmul!(C, A, B, α, β, M, K, N, ::Val{P}) where {P} = _call_loopmul!(C, A, B, α, β, M, K, N, Val{P}())

struct SyncMulFunc{TC,TA,TB,Α,Β,Md,Kd,Nd,AP,BCP,ID,TT,W₁,W₂,R₁,R₂} <: Function end
function (::SyncMulFunc{TC,TA,TB,Α,Β,Md,Kd,Nd,AP,BCP,ID,TT,W₁,W₂,R₁,R₂})(p::Ptr{UInt}) where {TC,TA,TB,Α,Β,Md,Kd,Nd,AP,BCP,ID,TT,W₁,W₂,R₁,R₂}
    offset, C = load(p, TC, 1)
    offset, A = load(p, TA, offset)
    offset, B = load(p, TB, offset)
    offset, α = load(p, Α, offset)
    offset, β = load(p, Β, offset)
    offset, M = load(p, Md, offset)
    offset, K = load(p, Kd, offset)
    offset, N = load(p, Nd, offset)
    offset, atomicp = load(p, AP, offset)
    offset, bcachep = load(p, BCP, offset)
    offset, id = load(p, ID, offset)
    offset, total_ids = load(p, TT, offset)
    sync_mul!(C, A, B, α, β, M, K, N, atomicp, bcachep, id, total_ids, StaticFloat{W₁}(), StaticFloat{W₂}(), StaticFloat{R₁}(), StaticFloat{R₂}())
    nothing
end

@generated function cfuncpointer(::T) where {T}
    precompile(T(), (Ptr{UInt},))
    quote
        $(Expr(:meta,:inline))
        @cfunction($(T()), Cvoid, (Ptr{UInt},))
    end
end

@inline function setup_matmul!(p::Ptr{UInt}, C::TC, A::TA, B::TB, α::Α, β::Β, M::Md, K::Kd, N::Nd, ::Val{P}) where {P,TC,TA,TB,Α,Β,Md,Kd,Nd}
    offset = store!(p, cfuncpointer(LoopMulFunc{P,TC,TA,TB,Α,Β,Md,Kd,Nd}()), 0)
    offset = store!(p, C, offset)
    offset = store!(p, A, offset)
    offset = store!(p, B, offset)
    offset = store!(p, α, offset)
    offset = store!(p, β, offset)
    offset = store!(p, M, offset)
    offset = store!(p, K, offset)
    offset = store!(p, N, offset)
    nothing
end

@inline function setup_syncmul!(
    p::Ptr{UInt}, C::TC, A::TA, B::TB, α::Α, β::Β, M::Md, K::Kd, N::Nd,
    ap::AP,bcp::BCP,id::ID,tt::TT,::StaticFloat{W₁},::StaticFloat{W₂},::StaticFloat{R₁},::StaticFloat{R₂}
) where {TC,TA,TB,Α,Β,Md,Kd,Nd,AP,BCP,ID,TT,W₁,W₂,R₁,R₂}
    offset = store!(p, cfuncpointer(SyncMulFunc{TC,TA,TB,Α,Β,Md,Kd,Nd,AP,BCP,ID,TT,W₁,W₂,R₁,R₂}()), 0)
    offset = store!(p, C, offset)
    offset = store!(p, A, offset)
    offset = store!(p, B, offset)
    offset = store!(p, α, offset)
    offset = store!(p, β, offset)
    offset = store!(p, M, offset)
    offset = store!(p, K, offset)
    offset = store!(p, N, offset)
    offset = store!(p, ap,  offset)
    offset = store!(p, bcp, offset)
    offset = store!(p, id,  offset)
    offset = store!(p, tt,  offset)
    nothing
end

function launch_thread_mul!(C, A, B, α, β, M, K, N, tid::Int, ::Val{P}) where {P}
    p = taskpointer(tid)
    while true
        if _atomic_cas_cmp!(p, SPIN, STUP)
            setup_matmul!(p, C, A, B, α, β, M, K, N, Val{P}())
            @assert _atomic_cas_cmp!(p, STUP, TASK)
            return
        elseif _atomic_cas_cmp!(p, WAIT, STUP)
            setup_matmul!(p, C, A, B, α, β, M, K, N, Val{P}())
            @assert _atomic_cas_cmp!(p, STUP, LOCK)
            wake_thread!(tid % UInt)
            return
        end
        pause()
    end
end
function launch_thread_mul!(
    C, A, B, α, β, M, K, N, ap, bcp, tid, tt,::StaticFloat{W₁},::StaticFloat{W₂},::StaticFloat{R₁},::StaticFloat{R₂}
) where {W₁,W₂,R₁,R₂}
    p = taskpointer(tid+one(UInt))
    while true
        if _atomic_cas_cmp!(p, SPIN, STUP)
            setup_syncmul!(
                p, C, A, B, α, β, M, K, N, ap, bcp, tid, tt,
                StaticFloat{W₁}(),StaticFloat{W₂}(),StaticFloat{R₁}(),StaticFloat{R₂}()
            )
            @assert _atomic_cas_cmp!(p, STUP, TASK)
            return
        elseif _atomic_cas_cmp!(p, WAIT, STUP)
            # we immediately write the `atomicp, bc, tid, total_tids` part, so we can dispatch to the same code as the other methods
            setup_syncmul!(
                p, C, A, B, α, β, M, K, N, ap, bcp, tid, tt,
                StaticFloat{W₁}(),StaticFloat{W₂}(),StaticFloat{R₁}(),StaticFloat{R₂}()
            )
            @assert _atomic_cas_cmp!(p, STUP, LOCK)
            wake_thread!(tid+one(UInt))
            return
        end
        pause()
    end
end


