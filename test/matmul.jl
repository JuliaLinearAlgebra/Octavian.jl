@time @testset "Matrix Multiply Float32" begin
    Mc, Kc, Nc = map(Int, Octavian.block_sizes(Float32))
    for logn ∈ range(log(1), log(1.5Nc+1), length = 5)
        n = round(Int, exp(logn))
        for logk ∈ range(log(1), log(1.5Kc+1), length = 5)
            k = round(Int, exp(logk))
            B = rand(Float32, k, n)
            B′ = permutedims(B)'
            for logm ∈ range(log(1), log(1.5Mc+1), length = 5)
                m = round(Int, exp(logm))
                A = rand(Float32, m, k)
                A′ = permutedims(A)'
                @show m, k, n
                @test @time(Octavian.matmul(A, B)) ≈ A * B
                @test @time(Octavian.matmul(A′, B)) ≈ A′ * B
                @test @time(Octavian.matmul(A, B′)) ≈ A * B′
                @test @time(Octavian.matmul(A′, B′)) ≈ A′ * B′
            end
        end
    end
end

@time @testset "Matrix Multiply Float64" begin
    Mc, Kc, Nc = map(Int, Octavian.block_sizes(Float64))
    for logn ∈ range(log(1), log(1.5Nc+1), length = 5)
        n = round(Int, exp(logn))
        for logk ∈ range(log(1), log(1.5Kc+1), length = 5)
            k = round(Int, exp(logk))
            B = rand(Float64, k, n)
            B′ = permutedims(B)'
            for logm ∈ range(log(1), log(1.5Mc+1), length = 5)
                m = round(Int, exp(logm))
                A = rand(Float64, m, k)
                A′ = permutedims(A)'
                @show m, k, n
                @test @time(Octavian.matmul(A, B)) ≈ A * B
                @test @time(Octavian.matmul(A′, B)) ≈ A′ * B
                @test @time(Octavian.matmul(A, B′)) ≈ A * B′
                @test @time(Octavian.matmul(A′, B′)) ≈ A′ * B′
            end
        end
    end
end

@time @testset "Matrix Multiply Int32" begin
    Mc, Kc, Nc = map(Int, Octavian.block_sizes(Int32))
    for logn ∈ range(log(1), log(1.5Nc+1), length = 5)
        n = round(Int, exp(logn))
        for logk ∈ range(log(1), log(1.5Kc+1), length = 5)
            k = round(Int, exp(logk))
            B = rand(Int32, k, n)
            B′ = permutedims(B)'
            for logm ∈ range(log(1), log(1.5Mc+1), length = 5)
                m = round(Int, exp(logm))
                A = rand(Int32, m, k)
                A′ = permutedims(A)'
                @show m, k, n
                @test @time(Octavian.matmul(A, B)) == A * B
                @test @time(Octavian.matmul(A′, B)) == A′ * B
                @test @time(Octavian.matmul(A, B′)) == A * B′
                @test @time(Octavian.matmul(A′, B′)) == A′ * B′
            end
        end
    end
end

@time @testset "Matrix Multiply Int64" begin
    Mc, Kc, Nc = map(Int, Octavian.block_sizes(Int64))
    for logn ∈ range(log(1), log(1.5Nc+1), length = 5)
        n = round(Int, exp(logn))
        for logk ∈ range(log(1), log(1.5Kc+1), length = 5)
            k = round(Int, exp(logk))
            B = rand(Int64, k, n)
            B′ = permutedims(B)'
            for logm ∈ range(log(1), log(1.5Mc+1), length = 5)
                m = round(Int, exp(logm))
                A = rand(Int64, m, k)
                A′ = permutedims(A)'
                @show m, k, n
                @test @time(Octavian.matmul(A, B)) == A * B
                @test @time(Octavian.matmul(A′, B)) == A′ * B
                @test @time(Octavian.matmul(A, B′)) == A * B′
                @test @time(Octavian.matmul(A′, B′)) == A′ * B′
            end
        end
    end
end
