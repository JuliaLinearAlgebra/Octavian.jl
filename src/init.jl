function __init__()
    init_bcache()
    nt = init_num_tasks()
    if nt < NUM_CORES && ("SUPPRESS_OCTAVIAN_WARNING" ∉ keys(ENV))
        msg = string(
            "Your system has $NUM_CORES physical cores, but `Octavian.jl` only has ",
            "$(nt > 1 ? "$(nt) threads" : "1 thread") available.",
            "For the best performance, you should start Julia with at least $(NUM_CORES) threads.",
            "",
        )
        @warn msg
    end
    reseet_bcache_lock!()
end

function init_bcache()
    resize!(BCACHE, SECOND_CACHE_SIZE * BCACHE_COUNT)
end

function init_num_tasks()
    num_tasks = _read_environment_num_tasks()::Int
    OCTAVIAN_NUM_TASKS[] = num_tasks
end

function _read_environment_num_tasks()
    environment_variable = get(ENV, "OCTAVIAN_NUM_TASKS", "")::String
    nt = min(Threads.nthreads(), VectorizationBase.NUM_CORES)::Int
    if isempty(environment_variable)
        return nt
    else
        return min(parse(Int, environment_variable)::Int, nt)
    end
end
