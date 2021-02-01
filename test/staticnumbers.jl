

@testset "Static Numbers" begin
    for i ∈ -10:10
        for j ∈ -10:10
            @test i+j == @inferred(Octavian.StaticInt(i) + Octavian.StaticFloat(j)) == @inferred(i + Octavian.StaticFloat(j)) == @inferred(Octavian.StaticFloat(i) + j) == @inferred(Octavian.StaticFloat(i) + Octavian.StaticInt(j)) == @inferred(Octavian.StaticFloat(i) + Octavian.StaticFloat(j))
            @test i-j == @inferred(Octavian.StaticInt(i) - Octavian.StaticFloat(j)) == @inferred(i - Octavian.StaticFloat(j)) == @inferred(Octavian.StaticFloat(i) - Octavian.StaticInt(j)) == @inferred(Octavian.StaticFloat(i) - j) == @inferred(Octavian.StaticFloat(i) - Octavian.StaticFloat(j))
            @test i*j == @inferred(Octavian.StaticInt(i) * Octavian.StaticFloat(j)) == @inferred(i * Octavian.StaticFloat(j)) == @inferred(Octavian.StaticFloat(i) * Octavian.StaticInt(j)) == @inferred(Octavian.StaticFloat(i) * j) == @inferred(Octavian.StaticFloat(i) * Octavian.StaticFloat(j))
            i == j == 0 && continue
            @test i/j == @inferred(Octavian.StaticInt(i) / Octavian.StaticFloat(j)) == @inferred(i / Octavian.StaticFloat(j)) == @inferred(Octavian.StaticFloat(i) / Octavian.StaticInt(j)) == @inferred(Octavian.StaticFloat(i) / j) == @inferred(Octavian.StaticFloat(i) / Octavian.StaticFloat(j))
        end
        if i ≥ 0
            @test sqrt(i) == @inferred(sqrt(Octavian.StaticInt(i))) == @inferred(sqrt(Octavian.StaticFloat(i))) == @inferred(sqrt(Octavian.StaticFloat(Float64(i))))
        end
    end
    @test Octavian.floortostaticint(1.0) === 1
    @test Octavian.floortostaticint(prevfloat(2.0)) === 1
    @test @inferred(Octavian.floortostaticint(Octavian.StaticFloat(1.0))) === Octavian.StaticInt(1)
    @test @inferred(Octavian.floortostaticint(Octavian.StaticFloat(prevfloat(2.0)))) === Octavian.StaticInt(1)

    @test Octavian.roundtostaticint(1.0) === 1
    @test Octavian.roundtostaticint(prevfloat(2.0)) === 2
    @test @inferred(Octavian.roundtostaticint(Octavian.StaticFloat(1.0))) === Octavian.StaticInt(1)
    @test @inferred(Octavian.roundtostaticint(Octavian.StaticFloat(prevfloat(2.0)))) === Octavian.StaticInt(2)
    @test @inferred(round(Octavian.StaticFloat(1.0))) === Octavian.StaticFloat(1)
    @test @inferred(round(Octavian.StaticFloat(prevfloat(2.0)))) === Octavian.StaticFloat(2)

    
end


