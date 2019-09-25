using Distributed
addprocs(9)

using Pkg.TOML

@everywhere begin 
    using Random: seed!
    using RMMMDNets
end

###

rmmmdnets_path = pathof(RMMMDNets) |> splitdir |> first |> splitdir |> first
hyper = TOML.parsefile("$rmmmdnets_path/examples/Hyper.toml")

function get_args_list_varying_D_z_and_Dg_h(dataset)
    Dg_h_list_dict = Dict(
        "gaussian" => ["10,10", "50,50", "100,100"],
        "ring" => ["20,20", "100,100", "200,200"],
    )
    args_list = []
    for model_name in [
        "gan", 
        "mmdnet", 
        "rmmmdnet",
    ], D_z in [2, 4, 8, 16], Dg_h in Dg_h_list_dict[dataset]
        args_dict = parse_toml(hyper, dataset, model_name)
        args = parse_args_dict(
            args_dict; 
            override=(D_z=D_z, Dg_h=Dg_h), 
            suffix="varying_D_z_and_Dg_h"
        )
        push!(args_list, args)
    end
    return args_list
end

function get_args_list_varying_D_fx()
    args_list = []
    for dataset in [
        "gaussian", 
        "ring"
    ], D_fx in [2]#, 4, 8, 16]
        args_dict = parse_toml(hyper, dataset, "rmmmdnet")
        args = parse_args_dict(
            args_dict; 
            override=(D_fx=D_fx,), 
            suffix="varying_D_fx"
        )
        push!(args_list, args)
    end
    return args_list
end

# Figure 1
args_list = get_args_list_varying_D_fx()

# Figure 2
# dataset = "gaussian"
# dataset = "ring"
# args_list = get_args_list_varying_D_z_and_Dg_h(dataset)

# Appendix: GRAM-net on MNIST
# args_list = [parse_args_dict(parse_toml(hyper, "mnist", "rmmmdnet"); override=(lr=1f-3, Df_h="conv", sigma="0.1,1,10,100",))]

###

@everywhere begin
    function run_exp(args)
        data = get_data(args.dataset)

        seed!(args.seed)

        model = get_model(args, data)
        evaluate(data, model)

        dataloader = DataLoader(data, args.batch_size)
        
        train!(model, args.n_epochs, dataloader)
        evaluate(data, model)

        model_fname = save!(model)

        return model, model_fname
    end
end

@sync @distributed for args in args_list
    run_exp(args)
end
