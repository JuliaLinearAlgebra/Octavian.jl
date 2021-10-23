
randdual(x, ::Val{N}=Val(3)) where {N} = ForwardDiff.Dual(x, ntuple(_ -> randn(), Val(N))...)
@time @testset "ForwardDiff.jl" begin
    m = 53
    n = 63
    k = 73

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

    @testset "real array from the left" begin
        config = ForwardDiff.JacobianConfig(nothing, C1, B1)
        I = LinearAlgebra.I(size(B1, 2))

        J1 = ForwardDiff.jacobian((C, B) -> Octavian.matmul!(C, A1, B), C1, B1, config)
        @test J1 ≈ kron(I, A1)

        J2 = ForwardDiff.jacobian((C, B) -> LinearAlgebra.mul!(C, A2, B), C2, B2, config)
        @test J1 ≈ kron(I, A2)
        @test J1 ≈ J2
    end

    @testset "real array from the right" begin
        # dense and column-major arrays
        config = ForwardDiff.JacobianConfig(nothing, C1, A1)

        J1 = ForwardDiff.jacobian((C, A) -> Octavian.matmul!(C, A, B1), C1, A1, config)
        J2 = ForwardDiff.jacobian((C, A) -> LinearAlgebra.mul!(C, A, B2), C2, A2, config)
        @test J1 ≈ J2

        # transposed arrays
        A1new = Matrix(A1')'
        A2new = Matrix(A2')'
        config = ForwardDiff.JacobianConfig(nothing, C1, A1new)

        J1 = ForwardDiff.jacobian((C, A) -> Octavian.matmul!(C, A, B1), C1, A1new, config)
        J2 = ForwardDiff.jacobian((C, A) -> LinearAlgebra.mul!(C, A, B2), C2, A2new, config)
        @test J1 ≈ J2

        # direct version using dual numbers
        A1dual = zeros(eltype(config), reverse(size(A1))...)
        A1dual .= A1'
        C1dual = zeros(eltype(config), size(C1)...)

        A2dual = deepcopy(A1dual)
        C2dual = deepcopy(C1dual)

        Octavian.matmul!(C1dual, A1dual', B1)
        Octavian.matmul!(C2dual, A2dual', B2)
        @test C1dual ≈ C2dual
    end

  @testset "two dual arrays" begin
    A1d = randdual.(A1)
    B1d = randdual.(B1)
    @test reinterpret(Float64, Octavian.matmul(A1d, B1d)) ≈ reinterpret(Float64, A1d * B1d)
    @test reinterpret(Float64, Octavian.matmul(@view(A1d[begin:end-1,:]), B1d)) ≈ reinterpret(Float64, @view(A1d[begin:end-1,:]) * B1d)
  end
end
