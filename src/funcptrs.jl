

struct LoopMulFunc{P,TC,TA,TB,Α,Β,Md,Kd,Nd} <: Function end
function (::LoopMulFunc{P,TC,TA,TB,Α,Β,Md,Kd,Nd})(p::Ptr{UInt}) where {P,TC,TA,TB,Α,Β,Md,Kd,Nd}
    offset, C = _atomic_load(p, TC, 1)
    offset, A = _atomic_load(p, TA, offset)
    offset, B = _atomic_load(p, TB, offset)
    offset, α = _atomic_load(p, Α, offset)
    offset, β = _atomic_load(p, Β, offset)
    offset, M = _atomic_load(p, Md, offset)
    offset, K = _atomic_load(p, Kd, offset)
    offset, N = _atomic_load(p, Nd, offset)
    _call_loopmul!(C, A, B, α, β, M, K, N, Val{P}())
    nothing
end
@inline _call_loopmul!(C, A, B, α, β, M, K, N, ::Val{false}) = loopmul!(C, A, B, α, β, M, K, N)
@inline function _call_loopmul!(C::StridedPointer{T}, A, B, α, β, M, K, N, ::Val{true}) where {T}
    if M*K < first_effective_cache(T) * R₂Default
        packaloopmul!(C, A, B, α, β, M, K, N)
        return
    else
        jmulpackAonly!(C, A, B, α, β, M, K, N, StaticFloat{W₁Default}(), StaticFloat{W₂Default}(), StaticFloat{R₁Default}(), StaticFloat{R₂Default}())
        return
    end
end
call_loopmul!(C, A, B, α, β, M, K, N, ::Val{P}) where {P} = _call_loopmul!(C, A, B, α, β, M, K, N, Val{P}())

struct SyncMulFunc{TC,TA,TB,Α,Β,Md,Kd,Nd,AP,BCP,ID,TT,W₁,W₂,R₁,R₂} <: Function end
function (::SyncMulFunc{TC,TA,TB,Α,Β,Md,Kd,Nd,AP,BCP,ID,TT,W₁,W₂,R₁,R₂})(p::Ptr{UInt}) where {TC,TA,TB,Α,Β,Md,Kd,Nd,AP,BCP,ID,TT,W₁,W₂,R₁,R₂}
    offset, C = _atomic_load(p, TC, 1)
    offset, A = _atomic_load(p, TA, offset)
    offset, B = _atomic_load(p, TB, offset)
    offset, α = _atomic_load(p, Α, offset)
    offset, β = _atomic_load(p, Β, offset)
    offset, M = _atomic_load(p, Md, offset)
    offset, K = _atomic_load(p, Kd, offset)
    offset, N = _atomic_load(p, Nd, offset)
    offset, atomicp = _atomic_load(p, AP, offset)
    offset, bcachep = _atomic_load(p, BCP, offset)
    offset, id = _atomic_load(p, ID, offset)
    offset, total_ids = _atomic_load(p, TT, offset)
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
    offset = _atomic_store!(p, cfuncpointer(LoopMulFunc{P,TC,TA,TB,Α,Β,Md,Kd,Nd}()), 0)
    offset = _atomic_store!(p, C, offset)
    offset = _atomic_store!(p, A, offset)
    offset = _atomic_store!(p, B, offset)
    offset = _atomic_store!(p, α, offset)
    offset = _atomic_store!(p, β, offset)
    offset = _atomic_store!(p, M, offset)
    offset = _atomic_store!(p, K, offset)
    offset = _atomic_store!(p, N, offset)
    nothing
end

@inline function setup_syncmul!(
    p::Ptr{UInt}, C::TC, A::TA, B::TB, α::Α, β::Β, M::Md, K::Kd, N::Nd,
    ap::AP,bcp::BCP,id::ID,tt::TT,::StaticFloat{W₁},::StaticFloat{W₂},::StaticFloat{R₁},::StaticFloat{R₂}
) where {TC,TA,TB,Α,Β,Md,Kd,Nd,AP,BCP,ID,TT,W₁,W₂,R₁,R₂}
    offset = _atomic_store!(p, cfuncpointer(SyncMulFunc{TC,TA,TB,Α,Β,Md,Kd,Nd,AP,BCP,ID,TT,W₁,W₂,R₁,R₂}()), 0)
    offset = _atomic_store!(p, C, offset)
    offset = _atomic_store!(p, A, offset)
    offset = _atomic_store!(p, B, offset)
    offset = _atomic_store!(p, α, offset)
    offset = _atomic_store!(p, β, offset)
    offset = _atomic_store!(p, M, offset)
    offset = _atomic_store!(p, K, offset)
    offset = _atomic_store!(p, N, offset)
    offset = _atomic_store!(p, ap,  offset)
    offset = _atomic_store!(p, bcp, offset)
    offset = _atomic_store!(p, id,  offset)
    offset = _atomic_store!(p, tt,  offset)
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
    p = taskpointer(tid)
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
            wake_thread!(tid)
            return
        end
        pause()
    end
end


