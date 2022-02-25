@testset "Aqua.jl" begin
  Aqua.test_all(Octavian, ambiguities=false)
  @test isempty(Test.detect_ambiguities(Octavian))
end
