function randdual(x)
  _x = zeros(HyperDualNumbers.Hyper{Float64}, size(x)...)
  for i in eachindex(x)
    _x[i] = HyperDualNumbers.Hyper(x[i], randn(), randn(), randn())
  end
  return _x
end

function reinterpretHD(T, A)
  tmp = reinterpret(T, A)
  return tmp[1:4:end, :]
end

@time @testset "HyperDualNumbers.jl" begin
  m = 53
  n = 63
  k = 73

  A1 = rand(Float64, m, k)
  B1 = rand(Float64, k, n)
  C1 = rand(Float64, m, n)

  α = Float64(2.0)
  β = Float64(2.0)

  @testset "real array from the right" begin
    A1dual = randdual(A1)
    C1dual = randdual(C1)

    A2dual = deepcopy(A1dual)
    B2 = deepcopy(B1)
    C2dual = deepcopy(C1dual)

    Octavian.matmul!(C1dual, A1dual, B1, α, β)
    LinearAlgebra.mul!(C2dual, A2dual, B2, α, β)
    @test reinterpret(Float64, C1dual) ≈ reinterpret(Float64, C2dual)
  end

  @testset "real array from the left" begin
    B1dual = randdual(B1)
    C1dual = randdual(C1)

    A2 = deepcopy(A1)
    B2dual = deepcopy(B1dual)
    C2dual = deepcopy(C1dual)

    Octavian.matmul!(C1dual, A1, B1dual, α, β)
    LinearAlgebra.mul!(C2dual, A2, B2dual, α, β)
    @test reinterpret(Float64, C1dual) ≈ reinterpret(Float64, C2dual)
  end

  @testset "transposed arrays" begin
    A1dual = randdual(A1')
    C1dual = randdual(C1)

    A2dual = deepcopy(A1dual)
    B2 = deepcopy(B1)
    C2dual = deepcopy(C1dual)

    C3dual = similar(C1dual)
    C4dual = similar(C2dual)
    Octavian.matmul!(C1dual, A1dual', B1)
    Octavian.matmul!(C2dual, A2dual', B2)
    Octavian.matmul_serial!(C3dual, A1dual', B1)
    Octavian.matmul_serial!(C4dual, A2dual', B2)

    Cref = zeros(Float64, size(C1)...)
    LinearAlgebra.mul!(Cref, A1, B1)
    @test (
      reinterpretHD(Float64, C1dual) ≈
      reinterpretHD(Float64, C2dual) ≈
      reinterpretHD(Float64, C3dual) ≈
      reinterpretHD(Float64, C4dual) ≈
      Cref
    ) && (
      reinterpret(Float64, C1dual) ≈
      reinterpret(Float64, C2dual) ≈
      reinterpret(Float64, C3dual) ≈
      reinterpret(Float64, C4dual)
    )
  end

  @testset "two dual arrays" begin
    A1d = randdual(A1)
    B1d = randdual(B1)
    @test reinterpret(Float64, Octavian.matmul(A1d, B1d, 1.3)) ≈
          reinterpret(Float64, Octavian.matmul_serial(A1d, B1d, 1.3)) ≈
          reinterpret(Float64, (A1d * B1d) .* 1.3)
    @test reinterpret(
            Float64,
            Octavian.matmul(@view(A1d[begin:end-1, :]), B1d)
          ) ≈
          reinterpret(
            Float64,
            Octavian.matmul_serial(@view(A1d[begin:end-1, :]), B1d)
          ) ≈
          reinterpret(Float64, @view(A1d[begin:end-1, :]) * B1d)
  end
end
