
using Octavian, StaticArrays, LinearAlgebra, BenchmarkTools, ProgressMeter

# BLAS.set_num_threads(1)

# For laptops that thermally throttle, you can set the `JULIA_SLEEP_BENCH` environment variable for #seconds to sleep before each `@belapsed`
const SLEEPTIME = parse(Float64, get(ENV, "JULIA_SLEEP_BENCH", "0"))
maybe_sleep() = iszero(SLEEPTIME) || sleep(SLEEPTIME)
# BenchmarkTools.DEFAULT_PARAMETERS.samples = 1_000_000
# BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

matrix_sizes(x::Integer) = (x, x, x)
matrix_sizes(x::NTuple{3}) = x

const matmulmethodnames =
  [:SMatrix, :MMatrix, :OctavianStatic, :OctavianDynamic];
function fill_bench_results!(br, lp, (M, K, N), t, i, j)
  name = matmulmethodnames[j]
  br[i, j, 1] = t
  gflops = 2e-9M * K * N / t
  br[i, j, 2] = gflops
  lp[j+1] = (name, gflops)
  nothing
end

function runbenches(sr, ::Type{T} = Float64) where {T}
  bench_results = Array{Float64}(undef, length(sr), 4, 2)
  p = Progress(length(sr))
  last_perfs = Vector{Tuple{Symbol,Union{Float64,NTuple{3,Int}}}}(
    undef,
    size(bench_results, 2) + 1
  )
  for (i, s) ∈ enumerate(sr)
    M, K, N = matrix_sizes(s)
    last_perfs[1] = (:Size, (M, K, N))
    Astatic = @SMatrix rand(T, M, K)
    Bstatic = @SMatrix rand(T, K, N)
    maybe_sleep()
    t = @belapsed $(Ref(Astatic))[] * $(Ref(Bstatic))[]
    fill_bench_results!(bench_results, last_perfs, (M, K, N), t, i, 1)
    Amutable = MArray(Astatic)
    Bmutable = MArray(Bstatic)
    Cmutable = MMatrix{M,N,T}(undef)
    maybe_sleep()
    t = @belapsed mul!($Cmutable, $Amutable, $Bmutable)
    fill_bench_results!(bench_results, last_perfs, (M, K, N), t, i, 2)
    Cmutable0 = copy(Cmutable)
    Cmutable .= NaN
    maybe_sleep()
    t = @belapsed matmul!($Cmutable, $Amutable, $Bmutable)
    fill_bench_results!(bench_results, last_perfs, (M, K, N), t, i, 3)
    A = Array(Amutable)
    B = Array(Bmutable)
    C = Array(Cmutable)
    maybe_sleep()
    t = @belapsed matmul!($C, $A, $B)
    fill_bench_results!(bench_results, last_perfs, (M, K, N), t, i, 4)
    @assert Array(Cmutable) ≈ Array(Cmutable0) ≈ C
    ProgressMeter.next!(p; showvalues = last_perfs)
  end
  bench_results
end

sizerange = 2:48
br = runbenches(sizerange);
using DataFrames, VegaLite

df = DataFrame(@view(br[:, :, 2]));
rename!(df, matmulmethodnames);
df.Size = sizerange

function pick_suffix(desc = "")
  suffix = if Bool(Octavian.VectorizationBase.has_feature(Val(:x86_64_avx512f)))
    "AVX512"
  elseif Bool(Octavian.VectorizationBase.has_feature(Val(:x86_64_avx2)))
    "AVX2"
  elseif Bool(Octavian.VectorizationBase.has_feature(Val(:x86_64_avx)))
    "AVX"
  else
    "REGSIZE$(Octavian.VectorizationBase.register_size())"
  end
  if desc != ""
    suffix *= '_' * desc
  end
  "$(Sys.CPU_NAME)_$suffix"
end

dfs = stack(
  df,
  matmulmethodnames;
  variable_name = :MatMulType,
  value_name = :GFLOPS
);
p =
  dfs |> @vlplot(
    :line,
    x = :Size,
    y = :GFLOPS,
    width = 900,
    height = 600,
    color = {:MatMulType}
  );
save(
  joinpath(
    pkgdir(Octavian),
    "docs/src/assets/sizedarraybenchmarks_$(pick_suffix()).svg"
  ),
  p
)
