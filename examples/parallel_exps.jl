using Distributed
addprocs(4)

using Pkg.TOML

@everywhere begin 
    using Random: seed!
    using RMMMDNets
end

###

# dataset = "gaussian"
dataset = "ring"
# dataset = "mnist"

model_name = "gan"
# model_name = "mmdnet"
# model_name = "rmmmdnet"

rmmmdnets_path = pathof(RMMMDNets) |> splitdir |> first |> splitdir |> first
hyper = TOML.parsefile("$rmmmdnets_path/examples/Hyper.toml")

args_dict = parse_toml(hyper, dataset, model_name)

args_list = [parse_args_dict(
    args_dict;
    override=(D_z=D_z,),
    suffix="test_toml"
) for D_z in [2, 4, 8, 16]]

###

@everywhere function run_exp(args)
    data = get_data(args.dataset)
    
    seed!(args.seed)

    model = get_model(args, data)

    evaluate(data, model)

    dataloader = DataLoader(data, args.batch_size)
    
    train!(model, args.n_epochs, dataloader)

    return data, model
end

@sync @distributed for args in args_list
    run_exp(args)
end
