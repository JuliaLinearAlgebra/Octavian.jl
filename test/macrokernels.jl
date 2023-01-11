@time @testset "Macrokernels" begin
  m = 20
  n = 30
  k = 40
  A1 = rand(Float64, m, k)
  B1 = rand(Float64, k, n)
  C1 = rand(Float64, m, n)
  A2 = deepcopy(A1)
  B2 = deepcopy(B1)
  C2 = deepcopy(C1)
  α = Float64(2.0)
  β = Float64(2.0)
  Octavian.loopmul!(
    VectorizationBase.zstridedpointer(C1),
    VectorizationBase.zstridedpointer(A1),
    VectorizationBase.zstridedpointer(B1),
    α,
    β,
    m,
    k,
    n
  )
  C2 = α * A2 * B2 + β * C2
  @test C1 ≈ C2
end
