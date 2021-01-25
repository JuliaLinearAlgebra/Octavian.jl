

using Octavian, VectorizationBase, ProgressMeter
using Octavian: StaticFloat
function matmul_pack_ab!(C, A, B, ::Val{W₁}, ::Val{W₂}, ::Val{R₁}, ::Val{R₂}) where {W₁, W₂, R₁, R₂}
    M, N = size(C); K = size(B,1)
    zc, za, zb = Octavian.zstridedpointer.((C,A,B))
    nspawn = min(Threads.nthreads(), VectorizationBase.num_cores())
    @elapsed(
        Octavian.matmul_pack_A_and_B!(
            zc, za, zb, StaticInt{1}(), StaticInt{0}(), M, K, N, nspawn,
            StaticFloat{W₁}(), StaticFloat{W₂}(), StaticFloat{R₁}(), StaticFloat{R₂}()
        )
    )
end

function bench_size(Cs, As, Bs, ::Val{W₁}, ::Val{W₂}, ::Val{R₁}, ::Val{R₂}) where {W₁, W₂, R₁, R₂}
    if length(first(Cs)) < length(last(Cs))
        matmul_pack_ab!(first(Cs), first(As), first(Bs), Val{W₁}(), Val{W₂}(), Val{R₁}(), Val{R₂}())
    else
        matmul_pack_ab!(last(Cs), last(As), last(Bs), Val{W₁}(), Val{W₂}(), Val{R₁}(), Val{R₂}())
    end
    gflop = 0.0
    for (C,A,B) ∈ zip(Cs,As,Bs)
        M, K, N = Octavian.matmul_sizes(C, A, B)
        # sleep(0.5)
        t = matmul_pack_ab!(C, A, B, Val{W₁}(), Val{W₂}(), Val{R₁}(), Val{R₂}())
        gf = 2e-9M*K*N / t
        gflop += gf
    end
    gflop / length(As)
end
matrix_sizes(s::Int) = (s,s,s)
matrix_sizes(MKN::NTuple{3,Int}) = MKN
size_range(l, u, len) = round.(Int, exp.(range(log(l), stop = log(u), length = len)))
function matrix_range(l, u, len, ::Type{T} = Float64) where {T}
    matrix_range(size_range(l, u, len), T)
end
function matrix_range(S, ::Type{T} = Float64) where {T}
    Alen = 0; Blen = 0; Clen = 0;
    for s ∈ S
        M, K, N = matrix_sizes(s)
        Alen = max(Alen, M*K)
        Blen = max(Blen, K*N)
        Clen = max(Clen, M*N)
    end
    Abuf = rand(T, Alen)
    Bbuf = rand(T, Blen)
    Cbuf = rand(T, Clen)
    As = Vector{Base.ReshapedArray{T, 2, SubArray{T, 1, Vector{T}, Tuple{Base.OneTo{Int}}, true}, Tuple{}}}(undef, length(S))
    Bs = similar(As); Cs = similar(As);
    for (i,s) ∈ enumerate(S)
        M, K, N = matrix_sizes(s)
        As[i] = reshape(view(Abuf, Base.OneTo(M * K)), (M, K))
        Bs[i] = reshape(view(Bbuf, Base.OneTo(K * N)), (K, N))
        Cs[i] = reshape(view(Cbuf, Base.OneTo(M * N)), (M, N))
    end
    Cs, As, Bs
end


T = Float64
min_size = round(Int, sqrt(0.65 * Octavian.VectorizationBase.cache_size(Val(3)) / sizeof(T)))
max_size = round(Int, sqrt( 32  * Octavian.VectorizationBase.cache_size(Val(3)) / sizeof(T)))

SR = size_range(max_size, min_size, 100);
const CsConst, AsConst, BsConst = matrix_range(SR, T);

function matmul_objective(params)
    print("Params: ", params, "; ")
    W₁, W₂, R₁, R₂ = params
    # print("(W₁ = $(round(W₁, sigdigits=4)); W₂ = $(round(W₂, sigdigits=4)); R₁ = $(round(R₁, sigdigits=4)); R₂ = $(round(R₂, sigdigits=4))); ")
    gflop = bench_size(CsConst, AsConst, BsConst, Val{W₁}(), Val{W₂}(), Val{R₁}(), Val{R₂}())
    println(gflop)
    - gflop
end

using Optim
hours = 60.0*60.0; days = 24hours;
init = Float64[Octavian.W₁Default(), Octavian.W₂Default(), Octavian.R₁Default(), Octavian.R₂Default()]

opt = Optim.optimize(
    matmul_objective, init, ParticleSwarm(lower = [0.001, 0.01, 0.3, 0.4], upper = [0.2, 2.0, 0.9, 0.99]),
    Optim.Options(iterations = 10^6, time_limit = 8hours)
);





