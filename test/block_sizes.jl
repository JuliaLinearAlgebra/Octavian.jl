@time @testset "block_sizes" begin
    @test Octavian._calculate_L3(1, 1, 1, Val(true)) == 0
    @test Octavian._calculate_L3(1, 1, 1, Val(false)) == 1
end
