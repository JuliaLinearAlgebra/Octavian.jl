using InteractiveUtils: versioninfo

versioninfo()

@info("Running tests with $(Threads.nthreads()) threads")

function is_coverage()
    return !iszero(Base.JLOptions().code_coverage)
end

const coverage = is_coverage()

@info("Code coverage is $(coverage ? "enabled" : "disabled")")

const run_all_tests = get(ENV, "RUN_ALL_TESTS", "false") == "true"

if run_all_tests
    @info("RUN_ALL_TESTS is $(run_all_tests)")
end
