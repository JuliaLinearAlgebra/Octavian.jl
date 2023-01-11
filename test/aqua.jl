@testset "Aqua.jl" begin
  Aqua.test_all(Octavian; ambiguities = false, project_toml_formatting = false)
  @test isempty(Test.detect_ambiguities(Octavian))
end
