
const OCTAVIAN_NUM_TASKS = Ref(1)
_nthreads() = OCTAVIAN_NUM_TASKS[]

@generated function calc_factors(
  ::Union{Val{nc},StaticInt{nc}} = num_cores()
) where {nc}
  t = Expr(:tuple)
  for i ∈ nc:-1:1
    d, r = divrem(nc, i)
    iszero(r) && push!(t.args, (i, d))
  end
  t
end
# const CORE_FACTORS = calc_factors()

MᵣW_mul_factor(::True) = StaticInt{4}()
MᵣW_mul_factor(::False) = StaticInt{9}()
MᵣW_mul_factor() = MᵣW_mul_factor(has_feature(Val(:x86_64_avx512f)))

W₁Default(::True) = StaticFloat64{0.0007423708195588264}()
W₂Default(::True) = StaticFloat64{0.7757548987718677}()
R₁Default(::True) = StaticFloat64{0.7936663315339363}()
R₂Default(::True) = StaticFloat64{0.7144577794375783}()

W₁Default_arch(::Val{:znver1}) = StaticFloat64{0.053918949422353986}()
W₂Default_arch(::Val{:znver1}) = StaticFloat64{0.3013238122374886}()
R₁Default_arch(::Val{:znver1}) = StaticFloat64{0.6077103834481342}()
R₂Default_arch(::Val{:znver1}) = StaticFloat64{0.8775382433240162}()

W₁Default_arch(::Union{Val{:znver2},Val{:znver3}}) = StaticFloat64{0.1}()
W₂Default_arch(::Union{Val{:znver2},Val{:znver3}}) =
  StaticFloat64{0.993489411720157}()
R₁Default_arch(::Union{Val{:znver2},Val{:znver3}}) =
  StaticFloat64{0.6052218809954467}()
R₂Default_arch(::Union{Val{:znver2},Val{:znver3}}) =
  StaticFloat64{0.7594052633561165}()

W₁Default_arch(_) = StaticFloat64{0.05846951331683783}()
W₂Default_arch(_) = StaticFloat64{0.16070447575367697}()
R₁Default_arch(_) = StaticFloat64{0.5370318382263098}()
R₂Default_arch(_) = StaticFloat64{0.5584398748982029}()

W₁Default(::False) = W₁Default_arch(VectorizationBase.cpu_name())
W₂Default(::False) = W₂Default_arch(VectorizationBase.cpu_name())
R₁Default(::False) = R₁Default_arch(VectorizationBase.cpu_name())
R₂Default(::False) = R₂Default_arch(VectorizationBase.cpu_name())

W₁Default() = W₁Default(has_feature(Val(:x86_64_avx512f)))
W₂Default() = W₂Default(has_feature(Val(:x86_64_avx512f)))
R₁Default() = R₁Default(has_feature(Val(:x86_64_avx512f)))
R₂Default() = R₂Default(has_feature(Val(:x86_64_avx512f)))

@static if Sys.ARCH === :x86_64 || Sys.ARCH === :i686
  first_cache() = StaticInt{2}()
else
  first_cache() = StaticInt{1}()
end

second_cache() = first_cache() + One()

_first_cache_size(fcs::StaticInt) = ifelse(
  eq(first_cache(), StaticInt(2)) & cache_inclusive(StaticInt(2)),
  fcs - cache_size(One()),
  fcs
)
_first_cache_size(::Nothing) = StaticInt(262144)
first_cache_size() = _first_cache_size(cache_size(first_cache()))

_second_cache_size(scs::StaticInt, ::True) = scs - cache_size(first_cache())
_second_cache_size(scs::StaticInt, ::False) = scs
_second_cache_size(::StaticInt{0}, ::Nothing) = StaticInt(3145728)
function second_cache_size()
  sc = second_cache()
  _second_cache_size(cache_size(sc), cache_inclusive(sc))
end

first_cache_size(::Val{T}) where {T} = first_cache_size() ÷ static_sizeof(T)
second_cache_size(::Val{T}) where {T} = second_cache_size() ÷ static_sizeof(T)

bcache_count() = VectorizationBase.num_cache(second_cache())

const BCACHEPTR = Ref{Ptr{Cvoid}}(C_NULL)
const BCACHE_LOCK = Threads.Atomic{UInt}(zero(UInt))

const ACACHEPTR = Ref{Ptr{Cvoid}}(C_NULL)
