@testset "Macros" begin
    a = Augustus.@_sync begin
        Augustus.@_spawn begin
            1 + 1
        end
    end
    if Threads.nthreads() > 1
        @test fetch(a) == 2
    else
        @test a == 2
    end
end
