function __init__()
    init_bcache()
    init_num_tasks()
    return nothing
end

function init_bcache()
    resize!(BCACHE, VectorizationBase.CACHE_SIZE[3] * VectorizationBase.CACHE_COUNT[3])
end

function init_num_tasks()
    num_tasks = _read_environment_num_tasks()::Int
    OCTAVIAN_NUM_TASKS[] = num_tasks
end

function _read_environment_num_tasks()
    environment_variable = get(ENV, "OCTAVIAN_NUM_TASKS", "")::String
    if isempty(environment_variable)
        return min(Threads.nthreads(), VectorizationBase.NUM_CORES)::Int
    else
        return parse(Int, environment_variable)::Int
    end
end
