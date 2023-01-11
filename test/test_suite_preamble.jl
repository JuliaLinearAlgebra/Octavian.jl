using InteractiveUtils: versioninfo

versioninfo()

@info("Sys.CPU_THREADS is $(Sys.CPU_THREADS)")
@info("Threads.nthreads() is $(Threads.nthreads()) threads")

is_coverage() = !iszero(Base.JLOptions().code_coverage)

const coverage = is_coverage()

@info("Code coverage is $(coverage ? "enabled" : "disabled")")
