
using Octavian, VectorizationBase
const F64 = Octavian.StaticFloat64
function matmul_pack_ab!(
  C,
  A,
  B,
  ::Val{W₁},
  ::Val{W₂},
  ::Val{R₁},
  ::Val{R₂}
) where {W₁,W₂,R₁,R₂}
  M, N = size(C)
  K = size(B, 1)
  zc, za, zb = Octavian.zstridedpointer.((C, A, B))
  nspawn = VectorizationBase.num_cores()
  nthreads = min(Int(nspawn), Threads.nthreads())
  # threads, torelease = Octavian.PolyesterWeave.__request_threads(
  #   (nspawn - 1) % UInt32,
  #   Octavian.PolyesterWeave.worker_pointer(),
  #   nothing
  # )
  t = Inf
  GC.@preserve C A B begin
    for _ ∈ 1:2
      t = min(
        t,
        @elapsed(
          Octavian.__matmul!(
            zc,
            za,
            zb,
            Octavian.One(),
            Octavian.Zero(),
            M,
            K,
            N,
            nthreads,
            F64(W₁),
            F64(W₂),
            F64(R₁),
            F64(R₂)
          )
        )
      )
    end
  end
  # Octavian.PolyesterWeave.free_threads!(torelease)
  return t
end

function bench_size(
  Cs,
  As,
  Bs,
  ::Val{W₁},
  ::Val{W₂},
  ::Val{R₁},
  ::Val{R₂}
) where {W₁,W₂,R₁,R₂}
  if length(first(Cs)) < length(last(Cs))
    matmul_pack_ab!(
      first(Cs),
      first(As),
      first(Bs),
      Val{W₁}(),
      Val{W₂}(),
      Val{R₁}(),
      Val{R₂}()
    )
  else
    matmul_pack_ab!(
      last(Cs),
      last(As),
      last(Bs),
      Val{W₁}(),
      Val{W₂}(),
      Val{R₁}(),
      Val{R₂}()
    )
  end
  repeat = 1
  gflop = 0.0
  for _ ∈ 1:repeat
    for (C, A, B) ∈ zip(Cs, As, Bs)
      M, K, N = Octavian.matmul_sizes(C, A, B)
      # sleep(0.5)
      t = matmul_pack_ab!(C, A, B, Val{W₁}(), Val{W₂}(), Val{R₁}(), Val{R₂}())
      gf = 2e-9M * K * N / t
      gflop += gf
    end
  end
  gflop / (repeat * length(As))
end
matrix_sizes(s::Int) = (s, s, s)
matrix_sizes(MKN::NTuple{3,Int}) = MKN
size_range(l, u, len) =
  round.(Int, exp.(range(log(l); stop = log(u), length = len)))
function matrix_range(l, u, len, ::Type{T} = Float64) where {T}
  matrix_range(size_range(l, u, len), T)
end
function matrix_range(S, ::Type{T} = Float64) where {T}
  Alen = 0
  Blen = 0
  Clen = 0
  for s ∈ S
    M, K, N = matrix_sizes(s)
    Alen = max(Alen, M * K)
    Blen = max(Blen, K * N)
    Clen = max(Clen, M * N)
  end
  Abuf = rand(T, Alen)
  Bbuf = rand(T, Blen)
  Cbuf = rand(T, Clen)
  As = Vector{
    Base.ReshapedArray{
      T,
      2,
      SubArray{T,1,Vector{T},Tuple{Base.OneTo{Int}},true},
      Tuple{}
    }
  }(
    undef,
    length(S)
  )
  Bs = similar(As)
  Cs = similar(As)
  for (i, s) ∈ enumerate(S)
    M, K, N = matrix_sizes(s)
    As[i] = reshape(view(Abuf, Base.OneTo(M * K)), (M, K))
    Bs[i] = reshape(view(Bbuf, Base.OneTo(K * N)), (K, N))
    Cs[i] = reshape(view(Cbuf, Base.OneTo(M * N)), (M, N))
  end
  Cs, As, Bs
end

T = Float32
min_size = min(
  round(
    Int,
    sqrt(
      (0.65 / 4) * Octavian.num_cores() * Octavian.second_cache_size() /
      sizeof(T)
    )
  ),
  2000
)
max_size = min(
  round(
    Int,
    sqrt(
      (32 / 4) * Octavian.num_cores() * Octavian.second_cache_size() /
      sizeof(T)
    )
  ),
  10_000
)

SR = size_range(max_size, min_size, 40);
const CsConst, AsConst, BsConst = matrix_range(SR, T);

# using Hyperopt
# ho = @hyperopt for i = 100, sampler=GPSampler(Max),
#     W₁ = LinRange(0.001, 0.3, 1000),
#     W₂ = LinRange(0.01, 2.0, 1000),
#     R₁ = LinRange(0.3, 0.9, 1000),
#     R₂ = LinRange(0.4, 0.99, 1000)
#     print("Params: ", (W₁, W₂, R₁, R₂), "; ")
#     gflop = bench_size(CsConst, AsConst, BsConst, Val{W₁}(), Val{W₂}(), Val{R₁}(), Val{R₂}())
#     println(gflop)
#     gflop
# end

# function restart(ho, iterations = 100)
#     olditer = ho.iterations
#     ho2 = Hyperoptimizer(
#         olditer + iterations,
#         ho.params,
#         ho.candidates,
#         ho.history[1:olditer],
#         ho.results[1:olditer],
#         ho.sampler#,
#         # ho.objective
#     )
#     # Hyperopt.optimize(ho2)
#     for nt ∈ ho2
#         (i, W₁, W₂, R₁, R₂) = nt
#         print("Params: ", (W₁, W₂, R₁, R₂), "; ")
#         gflop = bench_size(CsConst, AsConst, BsConst, Val{W₁}(), Val{W₂}(), Val{R₁}(), Val{R₂}())
#         println(gflop)
#         push!(ho2.results, gflop)
#     end
#     ho2
# end
# ho2 = restart(ho, 400)

function matmul_objective(params)
  print("Params= ", params, "; ")
  W₁, W₂, R₁, R₂ = params
  gflop = bench_size(
    CsConst,
    AsConst,
    BsConst,
    Val{W₁}(),
    Val{W₂}(),
    Val{R₁}(),
    Val{R₂}()
  )
  println(gflop)
  -gflop
end
using Optim
hours = 60.0 * 60.0;
days = 24hours;
init = Float64[
  Octavian.W₁Default(),
  Octavian.W₂Default(),
  Octavian.R₁Default(),
  Octavian.R₂Default()
]
lower = 0.25 .* init;
# upper = [1.25init[1], 1.25init[2], 0.75*init[3] + 0.25, 0.75*init[4] + 0.25];
upper = [0.9, 1.25init[2], 0.999, 0.999];
# init = [0.001, 0.9754033943603924, 0.5711159869399494, 0.7547361860432168];

opt = Optim.optimize(
  matmul_objective,
  init,
  ParticleSwarm(; lower = lower, upper = upper),
  Optim.Options(; iterations = 10^6, time_limit = 8 * hours)
);
