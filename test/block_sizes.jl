@time @testset "block_sizes" begin
  for T ∈ [Float32, Float64, Int16, Int32, Int64, UInt16, UInt32, UInt64]
    @inferred Octavian.block_sizes(Val(T))
    @inferred Octavian.first_cache_size(Val(T))
    @inferred Octavian.second_cache_size(Val(T))
  end

  @inferred Octavian.MᵣW_mul_factor()
  @inferred Octavian.W₁Default()
  @inferred Octavian.W₂Default()
  @inferred Octavian.R₁Default()
  @inferred Octavian.R₂Default()

  @inferred Octavian.first_cache()
  @inferred Octavian.second_cache()

  @inferred Octavian.first_cache_size()
  @inferred Octavian.second_cache_size()

  @inferred Octavian.bcache_count()
end
