
@testset "Matrix Multiply" begin

    for T ∈ (Float32, Float64, Int32, Int64)
        for logn ∈ range(log(1), log(4000), length = 5)
            n = round(Int, exp(logn))
            for logk ∈ range(log(1), log(4000), length = 5)
                k = round(Int, exp(logk))
                B = rand(T, k, n)
                B′ = permutedims(B)'
                for logm ∈ range(log(1), log(4000), length = 5)
                    m = round(Int, exp(logm))
                    A = rand(T, m, k)
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
end

