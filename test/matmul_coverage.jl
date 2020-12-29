@time @testset "Matrix Multiply Float32 (coverage)" begin
    m = 20
    n = 20
    k = 20
    A = rand(Float32, m, k)
    B = rand(Float32, k, n)
    A′ = permutedims(A)'
    B′ = permutedims(B)'
    @show m, k, n
    @test @time(Octavian.matmul(A, B)) ≈ A * B
    @test @time(Octavian.matmul(A′, B)) ≈ A′ * B
    @test @time(Octavian.matmul(A, B′)) ≈ A * B′
    @test @time(Octavian.matmul(A′, B′)) ≈ A′ * B′
end

@time @testset "Matrix Multiply Float64 (coverage)" begin
    m = 20
    n = 20
    k = 20
    A = rand(Float64, m, k)
    B = rand(Float64, k, n)
    A′ = permutedims(A)'
    B′ = permutedims(B)'
    @show m, k, n
    @test @time(Octavian.matmul(A, B)) ≈ A * B
    @test @time(Octavian.matmul(A′, B)) ≈ A′ * B
    @test @time(Octavian.matmul(A, B′)) ≈ A * B′
    @test @time(Octavian.matmul(A′, B′)) ≈ A′ * B′
end

@time @testset "Matrix Multiply Int32 (coverage)" begin
    m = 20
    n = 20
    k = 20
    A = rand(Int32, m, k)
    B = rand(Int32, k, n)
    A′ = permutedims(A)'
    B′ = permutedims(B)'
    @show m, k, n
    @test @time(Octavian.matmul(A, B)) == A * B
    @test @time(Octavian.matmul(A′, B)) == A′ * B
    @test @time(Octavian.matmul(A, B′)) == A * B′
    @test @time(Octavian.matmul(A′, B′)) == A′ * B′
end

@time @testset "Matrix Multiply Int64 (coverage)" begin
    m = 20
    n = 20
    k = 20
    A = rand(Int32, m, k)
    B = rand(Int32, k, n)
    A′ = permutedims(A)'
    B′ = permutedims(B)'
    @show m, k, n
    @test @time(Octavian.matmul(A, B)) == A * B
    @test @time(Octavian.matmul(A′, B)) == A′ * B
    @test @time(Octavian.matmul(A, B′)) == A * B′
    @test @time(Octavian.matmul(A′, B′)) == A′ * B′
end
