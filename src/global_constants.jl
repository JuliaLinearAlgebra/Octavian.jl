const BCACHE = UInt8[]

const OCTAVIAN_NUM_TASKS = Ref(1)
_nthreads() = OCTAVIAN_NUM_TASKS[]

@generated function calc_factors(::Val{nc} = Val{NUM_CORES}()) where {nc}
    t = Expr(:tuple)
    for i ∈ nc:-1:1
        d, r = divrem(nc, i)
        iszero(r) && push!(t.args, (i, d))
    end
    t
end
const CORE_FACTORS = calc_factors()

const MᵣW_mul_factor = VectorizationBase.REGISTER_SIZE === 64 ? StaticInt{4}() : StaticInt{9}()

if VectorizationBase.AVX512F
    const W₁Default = 0.006131471369820045
    const W₂Default = 0.7646526105725088
    const R₁Default = 0.5577652012322807
    const R₂Default = 0.7586696322536083
else
    const W₁Default = 0.0888571100241128
    const W₂Default = 0.5283378068764165
    const R₁Default = 0.41520001574995036
    const R₂Default = 0.681381024297185
end

