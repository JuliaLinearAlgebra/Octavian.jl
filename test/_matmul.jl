# The following variables need to be defined before `include`-ing this file:
# `testset_name_suffix`
# `n_values`
# `k_values`
# `m_values`
# `additional_n_k_m_values`

# If you don't want any additional values, just set:
# additional_n_k_m_values = []

n_k_m_product = Iterators.product(n_values, k_values, m_values)
all_n_k_m_values = Iterators.flatten([n_k_m_product, additional_n_k_m_values])

@time @testset "Matrix Multiply Float32 $(testset_name_suffix)" begin
    T = Float32
    for n_k_m ∈ all_n_k_m_values
        n, k, m = n_k_m
        A = rand(T, m, k)
        B = rand(T, k, n)
        A′ = permutedims(A)'
        B′ = permutedims(B)'
        AB = A * B; A′B = A′ * B; AB′ = A * B′; A′B′ = A′ * B′;
        @info "" T n k m
        @test @time(Octavian.matmul(A, B)) ≈ AB
        @test @time(Octavian.matmul(A′, B)) ≈ A′B
        @test @time(Octavian.matmul(A, B′)) ≈ AB′
        @test @time(Octavian.matmul(A′, B′)) ≈ A′B′
        @test @time(Octavian.matmul_serial(A, B)) ≈ AB
        @test @time(Octavian.matmul_serial(A′, B)) ≈ A′B
        @test @time(Octavian.matmul_serial(A, B′)) ≈ AB′
        @test @time(Octavian.matmul_serial(A′, B′)) ≈ A′B′
    end
end

@time @testset "Matrix Multiply Float64 $(testset_name_suffix)" begin
    T = Float64
    for n_k_m ∈ all_n_k_m_values
        n, k, m = n_k_m
        A = rand(T, m, k)
        B = rand(T, k, n)
        A′ = permutedims(A)'
        B′ = permutedims(B)'
        AB = A * B; A′B = A′ * B; AB′ = A * B′; A′B′ = A′ * B′;
        @info "" T n k m
        @test @time(Octavian.matmul(A, B)) ≈ AB
        @test @time(Octavian.matmul(A′, B)) ≈ A′B
        @test @time(Octavian.matmul(A, B′)) ≈ AB′
        @test @time(Octavian.matmul(A′, B′)) ≈ A′B′
        @test @time(Octavian.matmul_serial(A, B)) ≈ AB
        @test @time(Octavian.matmul_serial(A′, B)) ≈ A′B
        @test @time(Octavian.matmul_serial(A, B′)) ≈ AB′
        @test @time(Octavian.matmul_serial(A′, B′)) ≈ A′B′
    end
end

@time @testset "Matrix Multiply Int32 $(testset_name_suffix)" begin
    T = Int32
    for n_k_m ∈ all_n_k_m_values
        n, k, m = n_k_m
        A = rand(T, m, k)
        B = rand(T, k, n)
        A′ = permutedims(A)'
        B′ = permutedims(B)'
        AB = A * B; A′B = A′ * B; AB′ = A * B′; A′B′ = A′ * B′;
        @info "" T n k m
        @test @time(Octavian.matmul(A, B)) == AB
        @test @time(Octavian.matmul(A′, B)) == A′B
        @test @time(Octavian.matmul(A, B′)) == AB′
        @test @time(Octavian.matmul(A′, B′)) == A′B′
        @test @time(Octavian.matmul_serial(A, B)) == AB
        @test @time(Octavian.matmul_serial(A′, B)) == A′B
        @test @time(Octavian.matmul_serial(A, B′)) == AB′
        @test @time(Octavian.matmul_serial(A′, B′)) == A′B′
    end
end

@time @testset "Matrix Multiply Int64 $(testset_name_suffix)" begin
    T = Int64
    for n_k_m ∈ all_n_k_m_values
        n, k, m = n_k_m
        A = rand(T, m, k)
        B = rand(T, k, n)
        A′ = permutedims(A)'
        B′ = permutedims(B)'
        AB = A * B; A′B = A′ * B; AB′ = A * B′; A′B′ = A′ * B′;
        @info "" T n k m
        @test @time(Octavian.matmul(A, B)) == AB
        @test @time(Octavian.matmul(A′, B)) == A′B
        @test @time(Octavian.matmul(A, B′)) == AB′
        @test @time(Octavian.matmul(A′, B′)) == A′B′
        @test @time(Octavian.matmul_serial(A, B)) == AB
        @test @time(Octavian.matmul_serial(A′, B)) == A′B
        @test @time(Octavian.matmul_serial(A, B′)) == AB′
        @test @time(Octavian.matmul_serial(A′, B′)) == A′B′
    end
end
