@testset "init" begin
    withenv("OCTAVIAN_NUM_TASKS" => "") do
        @test Octavian._read_environment_num_tasks() == min(Threads.nthreads(), VectorizationBase.NUM_CORES)
    end
    withenv("OCTAVIAN_NUM_TASKS" => "99") do
        @test Octavian._read_environment_num_tasks() == 99
    end
end
