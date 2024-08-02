@testset "Aqua.jl" begin
    Aqua.test_all(Octavian; deps_compat=false, ambiguities=false)
    Aqua.test_ambiguities(Octavian; recursive=false)
end
