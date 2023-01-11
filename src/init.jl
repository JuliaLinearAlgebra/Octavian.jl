function __init__()
  @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" include(
    "forward_diff.jl"
  )

  init_acache()
  init_bcache()
  nt = init_num_tasks()
  if nt < num_cores() && ("OCTAVIAN_WARNING" ∈ keys(ENV))
    msg = string(
      "Your system has $(num_cores()) physical cores, but `Octavian.jl` only has ",
      "$(nt > 1 ? "$(nt) threads" : "$(nt) thread") available. ",
      "For the best performance, you should start Julia with at least $(num_cores()) threads."
    )
    @warn msg
  end
  reset_bcache_lock!()
end

function init_bcache()
  if bcache_count() ≢ Zero()
    if BCACHEPTR[] == C_NULL
      BCACHEPTR[] = VectorizationBase.valloc(
        Threads.nthreads() * second_cache_size() * bcache_count(),
        Cvoid,
        ccall(:jl_getpagesize, Int, ())
      )
    end
  end
  nothing
end

function init_acache()
  if ACACHEPTR[] == C_NULL
    ACACHEPTR[] = VectorizationBase.valloc(
      first_cache_size() * init_num_tasks(),
      Cvoid,
      ccall(:jl_getpagesize, Int, ())
    )
  end
  nothing
end

function init_num_tasks()
  num_tasks = _read_environment_num_tasks()
  OCTAVIAN_NUM_TASKS[] = num_tasks
end

function _read_environment_num_tasks()::Int
  environment_variable = get(ENV, "OCTAVIAN_NUM_TASKS", "")::String
  nt = min(Threads.nthreads(), VectorizationBase.num_cores())::Int
  if isempty(environment_variable)
    return nt
  else
    return min(parse(Int, environment_variable)::Int, nt)
  end
end
