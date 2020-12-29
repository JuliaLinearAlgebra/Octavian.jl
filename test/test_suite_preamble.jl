using InteractiveUtils: versioninfo

versioninfo()

@info("Running tests with $(Threads.nthreads()) threads")

function is_coverage()
    return !iszero(Base.JLOptions().code_coverage)
end

const coverage = is_coverage()

@info("Code coverage is $(coverage ? "enabled" : "disabled")")
