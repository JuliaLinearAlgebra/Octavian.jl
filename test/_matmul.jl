# The following variables need to be defined before `include`-ing this file:
# `testset_name_suffix`
# `n_values`
# `k_values`
# `m_values`
for T ∈ (ComplexF32, ComplexF64, Complex{Int}, Complex{Int32})
    @time @testset "Matrix Multiply $T $(testset_name_suffix)" begin
        for n ∈ n_values
            for k ∈ k_values
                for m ∈ m_values
                    A = rand(T, m, k)
                    B = rand(T, k, n)

                    Are = real.(A)
                    Bre = real.(B)
                    
                    A′ = permutedims(A)'
                    B′ = permutedims(B)'
                    AB = A * B;
                    A′B = A′*B
                    AB′ = A*B′
                    A′B′= A′*B′

                    AreB = Are*B
                    ABre = A*Bre
                    
                    @info "" T n k m
                    @test @time(Octavian.matmul(A, B)) ≈ AB
                    @test @time(Octavian.matmul(A, Bre)) ≈ ABre
                    @test @time(Octavian.matmul(Are, B)) ≈ AreB
                    @test @time(Octavian.matmul(A′, B)) ≈ A′B
                    @test @time(Octavian.matmul(A, B′)) ≈ AB′
                    @test @time(Octavian.matmul(A′, B′)) ≈ A′B′

                    
                    @test @time(Octavian.matmul_serial(A, B)) ≈ AB
                    @test @time(Octavian.matmul_serial(A, Bre)) ≈ ABre
                    @test @time(Octavian.matmul_serial(Are, B)) ≈ AreB
                    @test @time(Octavian.matmul_serial(A′, B)) ≈ A′B
                    @test @time(Octavian.matmul_serial(A, B′)) ≈ AB′
                    @test @time(Octavian.matmul_serial(A′, B′)) ≈ A′B′
                    
                    C = Matrix{T}(undef, n, m)'
                    @test @time(Octavian.matmul!(C, A, B)) ≈ AB

                    C1 = rand(T, m, n)
                    C2 = copy(C1)
                    α, β = T(1 - 2im), T(3 + 4im)
                    @test @time(Octavian.matmul!(C1, A, B, α, β)) ≈ Octavian.matmul!(C2, A, B, α, β)
                end
            end
        end
    end
    testset_name_suffix === "(coverage)" && break
end

@time @testset "Matrix Multiply Float64 $(testset_name_suffix)" begin
    T = Float64
    for n ∈ n_values
        for k ∈ k_values
            for m ∈ m_values
                A = rand(T, m, k)
                B = rand(T, k, n)
                A′ = permutedims(A)'
                B′ = permutedims(B)'
                AB = A * B;
                @info "" T n k m
                @test @time(Octavian.matmul(A, B)) ≈ AB
                @test @time(Octavian.matmul(A′, B)) ≈ AB
                @test @time(Octavian.matmul(A, B′)) ≈ AB
                @test @time(Octavian.matmul(A′, B′)) ≈ AB
                @test @time(Octavian.matmul_serial(A, B)) ≈ AB
                @test @time(Octavian.matmul_serial(A′, B)) ≈ AB
                @test @time(Octavian.matmul_serial(A, B′)) ≈ AB
                @test @time(Octavian.matmul_serial(A′, B′)) ≈ AB
            end
        end
    end
    m = k = n = max(8Octavian.OCTAVIAN_NUM_TASKS[], 400)
    A = rand(T, m, k);
    B = rand(T, k, n);
    A′ = permutedims(A)';
    B′ = permutedims(B)';
    AB = A * B;
    @test matmul_pack_ab!(similar(AB), A, B) ≈ AB
    @test matmul_pack_ab!(similar(AB), A, B′) ≈ AB
    @test matmul_pack_ab!(similar(AB), A′, B) ≈ AB
    @test matmul_pack_ab!(similar(AB), A′, B′) ≈ AB
end

@time @testset "Matrix Multiply Float32 $(testset_name_suffix)" begin
    T = Float32
    for n ∈ n_values
        for k ∈ k_values
            for m ∈ m_values
                A = rand(T, m, k)
                B = rand(T, k, n)
                A′ = permutedims(A)'
                B′ = permutedims(B)'
                AB = A * B;
                @info "" T n k m
                @test @time(Octavian.matmul(A, B)) ≈ AB
                @test @time(Octavian.matmul(A′, B)) ≈ AB
                @test @time(Octavian.matmul(A, B′)) ≈ AB
                @test @time(Octavian.matmul(A′, B′)) ≈ AB
                @test @time(Octavian.matmul_serial(A, B)) ≈ AB
                @test @time(Octavian.matmul_serial(A′, B)) ≈ AB
                @test @time(Octavian.matmul_serial(A, B′)) ≈ AB
                @test @time(Octavian.matmul_serial(A′, B′)) ≈ AB
            end
        end
    end
    m = k = n = max(8Octavian.OCTAVIAN_NUM_TASKS[], 400)
    A = rand(T, m, k);
    B = rand(T, k, n);
    A′ = permutedims(A)';
    B′ = permutedims(B)';
    AB = A * B;
    @test matmul_pack_ab!(similar(AB), A, B) ≈ AB
    @test matmul_pack_ab!(similar(AB), A, B′) ≈ AB
    @test matmul_pack_ab!(similar(AB), A′, B) ≈ AB
    @test matmul_pack_ab!(similar(AB), A′, B′) ≈ AB
end

@time @testset "Matrix Multiply Int32 $(testset_name_suffix)" begin
    T = Int32
    for n ∈ n_values
        for k ∈ k_values
            for m ∈ m_values
                A = rand(T, m, k)
                B = rand(T, k, n)
                A′ = permutedims(A)'
                B′ = permutedims(B)'
                AB = A * B;
                @info "" T n k m
                @test @time(Octavian.matmul(A, B)) ≈ AB
                @test @time(Octavian.matmul(A′, B)) ≈ AB
                @test @time(Octavian.matmul(A, B′)) ≈ AB
                @test @time(Octavian.matmul(A′, B′)) ≈ AB
                @test @time(Octavian.matmul_serial(A, B)) ≈ AB
                @test @time(Octavian.matmul_serial(A′, B)) ≈ AB
                @test @time(Octavian.matmul_serial(A, B′)) ≈ AB
                @test @time(Octavian.matmul_serial(A′, B′)) ≈ AB
            end
        end
    end
    m = k = n = max(8Octavian.OCTAVIAN_NUM_TASKS[], 400)
    A = rand(T, m, k);
    B = rand(T, k, n);
    A′ = permutedims(A)';
    B′ = permutedims(B)';
    AB = A * B;
    @test matmul_pack_ab!(similar(AB), A, B) ≈ AB
    @test matmul_pack_ab!(similar(AB), A, B′) ≈ AB
    @test matmul_pack_ab!(similar(AB), A′, B) ≈ AB
    @test matmul_pack_ab!(similar(AB), A′, B′) ≈ AB
end

@time @testset "Matrix Multiply Int64 $(testset_name_suffix)" begin
    T = Int64
    for n ∈ n_values
        for k ∈ k_values
            for m ∈ m_values
                A = rand(T, m, k)
                B = rand(T, k, n)
                A′ = permutedims(A)'
                B′ = permutedims(B)'
                AB = A * B;
                @info "" T n k m
                @test @time(Octavian.matmul(A, B)) ≈ AB
                @test @time(Octavian.matmul(A′, B)) ≈ AB
                @test @time(Octavian.matmul(A, B′)) ≈ AB
                @test @time(Octavian.matmul(A′, B′)) ≈ AB
                @test @time(Octavian.matmul_serial(A, B)) ≈ AB
                @test @time(Octavian.matmul_serial(A′, B)) ≈ AB
                @test @time(Octavian.matmul_serial(A, B′)) ≈ AB
                @test @time(Octavian.matmul_serial(A′, B′)) ≈ AB
            end
        end
    end
    m = k = n = max(8Octavian.OCTAVIAN_NUM_TASKS[], 400)
    A = rand(T, m, k);
    B = rand(T, k, n);
    A′ = permutedims(A)';
    B′ = permutedims(B)';
    AB = A * B;
    @test matmul_pack_ab!(similar(AB), A, B) ≈ AB
    @test matmul_pack_ab!(similar(AB), A, B′) ≈ AB
    @test matmul_pack_ab!(similar(AB), A′, B) ≈ AB
    @test matmul_pack_ab!(similar(AB), A′, B′) ≈ AB
end

@time @testset "zero-sized-matrices" begin
    @test Octavian.matmul_serial(randn(0,0), randn(0,0)) == zeros(0, 0)
    @test Octavian.matmul_serial(randn(2,3), randn(3,0)) == zeros(2, 0)
    @test Octavian.matmul_serial(randn(2,0), randn(0,2)) == zeros(2, 2)
    @test Octavian.matmul_serial!(ones(2,2),randn(2,0), randn(0,2), 1.0, 2.0) == ones(2, 2) .* 2
    @test Octavian.matmul(randn(0,0), randn(0,0)) == zeros(0, 0)
    @test Octavian.matmul(randn(2,3), randn(3,0)) == zeros(2, 0)
    @test Octavian.matmul(randn(2,0), randn(0,2)) == zeros(2, 2)
    @test Octavian.matmul!(ones(2,2),randn(2,0), randn(0,2), 1.0, 2.0) == ones(2, 2) .* 2
end
