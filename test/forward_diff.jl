
@time @testset "ForwardDiff.jl" begin
  m = 5
  n = 6
  k = 7

  A1 = rand(Float64, m, k)
  B1 = rand(Float64, k, n)
  C1 = rand(Float64, m, n)

  A2 = deepcopy(A1)
  B2 = deepcopy(B1)
  C2 = deepcopy(C1)

  α = Float64(2.0)
  β = Float64(2.0)

  Octavian.matmul!(C1, A1, B1, α, β)
  LinearAlgebra.mul!(C2, A2, B2, α, β)
  @test C1 ≈ C2

  # real array from the left
  config = ForwardDiff.JacobianConfig(nothing, C1, B1)
  I = LinearAlgebra.I(size(B1, 2))

  J1 = ForwardDiff.jacobian((C, B) -> Octavian.matmul!(C, A1, B), C1, B1, config)
  @test J1 ≈ kron(I, A1)

  J2 = ForwardDiff.jacobian((C, B) -> LinearAlgebra.mul!(C, A2, B), C2, B2, config)
  @test J1 ≈ kron(I, A2)
  @test J1 ≈ J2

  # real array from the right
  config = ForwardDiff.JacobianConfig(nothing, C1, A1)

  J1 = ForwardDiff.jacobian((C, A) -> Octavian.matmul!(C, A, B1), C1, A1, config)
  J2 = ForwardDiff.jacobian((C, A) -> LinearAlgebra.mul!(C, A, B2), C2, A2, config)
  @test J1 ≈ J2
end
