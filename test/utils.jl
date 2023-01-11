@time @testset "utils" begin
  @testset "check_sizes" begin
    a = Octavian.StaticInt{1}()
    b = Octavian.StaticInt{2}()
    @test Octavian._select(a, a) === a
    @test Octavian._select(a, 1) === a
    @test Octavian._select(1, a) === a
    @test Octavian._select(1, 1) === 1
    @test_throws AssertionError Octavian.matmul_sizes(
      rand(3, 2),
      rand(3, 4),
      rand(5, 2)
    )
    @test_throws AssertionError Octavian.matmul_sizes(
      rand(3, 2),
      rand(3, 4),
      rand(4, 5)
    )
    @test_throws AssertionError Octavian.matmul_sizes(
      rand(3, 2),
      rand(5, 4),
      rand(4, 2)
    )
  end
end
