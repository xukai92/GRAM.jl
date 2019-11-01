using Distributed
addprocs(9)

@everywhere using RMMMDNets

###

grampath = pathof(RMMMDNets) |> splitdir |> first |> splitdir |> first
hyperpath = "$grampath/examples/Hyper.toml"

function get_args_list_varying_Dz_and_Dhs_g(dataset)
    Dhs_g_list_dict = Dict(
        "gaussian" => ["10,10", "50,50", "100,100"],
        "ring" => ["20,20", "100,100", "200,200"],
    )
    args_list = []
    for modelname in [
        "gan", 
        "mmdnet", 
        "rmmmdnet",
    ], Dz in [2, 4, 8, 16], Dhs_g in Dhs_g_list_dict[dataset]
        argdict = parsetoml(hyperpath, dataset, modelname)
        args = parseargdict(
            argdict; 
            override=(Dz=Dz, Dhs_g=Dhs_g), 
            suffix="varying_Dz_and_Dhs_g"
        )
        push!(args_list, args)
    end
    return args_list
end

function get_args_list_varying_Df()
    args_list = []
    for dataset in [
        "gaussian", 
        "ring"
    ], Df in [2]#, 4, 8, 16]
        argdict = parsetoml(hyperpath, dataset, "rmmmdnet")
        args = parseargdict(
            argdict; 
            override=(Df=Df,), 
            suffix="varying_Df"
        )
        push!(args_list, args)
    end
    return args_list
end

# Figure 1
args_list = get_args_list_varying_Df()

# Figure 2
# dataset = "gaussian"
# dataset = "ring"
# args_list = get_args_list_varying_Dz_and_Dhs_g(dataset)

# Appendix: GRAM-net on MNIST
# args_list = [parseargdict(parsetoml(hyperpath, "mnist", "rmmmdnet"); override=(lr=1f-3, Df_h="conv", sigma="0.1,1,10,100",))]

###

@sync @distributed for args in args_list
    run_exp(args)
end
