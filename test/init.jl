@testset "init" begin
    withenv("OCTAVIAN_NUM_TASKS" => "") do
        @test Octavian._read_environment_num_tasks() == min(Threads.nthreads(), VectorizationBase.num_cores())
    end
    withenv("OCTAVIAN_NUM_TASKS" => "1") do
        @test Octavian._read_environment_num_tasks() == 1
    end
    withenv("OCTAVIAN_NUM_TASKS" => "99") do
        @test Octavian._read_environment_num_tasks() == min(Threads.nthreads(), VectorizationBase.num_cores())
    end
end
