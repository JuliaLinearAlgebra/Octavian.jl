@time @testset "utils" begin
    @testset "check_sizes" begin
        a = Octavian.StaticInt{1}()
        b = Octavian.StaticInt{2}()
        @test Octavian.check_sizes(a, a) == a
        @test_throws ErrorException Octavian.check_sizes(a, b)
        @test Octavian.check_sizes(a, 1) == a
        @test_throws AssertionError Octavian.check_sizes(a, 100)
        @test Octavian.check_sizes(2, b) == b
        @test_throws AssertionError Octavian.check_sizes(200, b)
    end
end
