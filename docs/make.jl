using Octavian
using Documenter

makedocs(;
    modules=[Octavian],
    authors="Mason Protter, Chris Elrod, Dilum Aluthge, and contributors",
    repo="https://github.com/JuliaLinearAlgebra/Octavian.jl/blob/{commit}{path}#L{line}",
    sitename="Octavian.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaLinearAlgebra.github.io/Octavian.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Public API" => "public-api.md",
        "Internals (Private)" => "internals.md",
    ],
    strict=true,
)

deploydocs(;
    repo="github.com/JuliaLinearAlgebra/Octavian.jl",
)
