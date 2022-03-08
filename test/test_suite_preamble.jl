using InteractiveUtils: versioninfo

versioninfo()

@info("Sys.CPU_THREADS is $(Sys.CPU_THREADS)")
@info("Threads.nthreads() is $(Threads.nthreads()) threads")

if Threads.nthreads() > 1 && Sys.CPU_THREADS > 1 && CPUSummary.num_cores() == 1
    CPUSummary.num_cores() = CPUSummary.static(2)
end

function is_coverage()
    return !iszero(Base.JLOptions().code_coverage)
end

const coverage = is_coverage()

@info("Code coverage is $(coverage ? "enabled" : "disabled")")
