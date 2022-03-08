using InteractiveUtils: versioninfo

versioninfo()

@info("Sys.CPU_THREADS is $(Sys.CPU_THREADS)")
@info("Threads.nthreads() is $(Threads.nthreads()) threads")

function is_coverage()
    return !iszero(Base.JLOptions().code_coverage)
end

const coverage = is_coverage()

@info("Code coverage is $(coverage ? "enabled" : "disabled")")
