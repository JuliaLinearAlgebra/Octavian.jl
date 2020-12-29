@time Test.@testset "PointerMatrix" begin
    mem = Octavian.L2Buffer(Float64);
    ptr = Base.unsafe_convert(Ptr{Float64}, mem)
    block = Octavian.PointerMatrix(ptr, (10, 20))
    Test.@test Base.unsafe_convert(Ptr{Float64}, block) == pointer(block.p)
    GC.@preserve mem begin
        block[1] = 2.3
        Test.@test block[1] == 2.3
        block[4, 5] = 67.89
        Test.@test block[4, 5] == 67.89
    end
end
