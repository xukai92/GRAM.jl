using Distributed
addprocs(4)

using MLToolkit: flatten_dict, dict2namedtuple

@everywhere begin 
    using Random: seed!
    using RMMMDNets
end

###

rmmmdnets_path = pathof(RMMMDNets) |> splitdir |> first |> splitdir |> first
include("$rmmmdnets_path/scripting.jl")

# dataset = "gaussian"
dataset = "ring"
# dataset = "mnist"

# model_name = "mmdnet"
model_name = "rmmmdnet"

args_list = [get_args(
    dataset, 
    model_name; 
    override=Dict(:seed => seed), 
    suffix="seed=$seed-bugfixing"
) for seed in [1, 2, 3, 4]]

###

@everywhere function run_exp(args)
    data = get_data(args.dataset)
    
    seed!(args.seed)

    model = get_model(args, data)

    vis(data, model)

    dataloader = DataLoader(data, args.batch_size)
    
    train!(model, args.n_epochs, dataloader)
end

@sync @distributed for args in args_list
    run_exp(args)
end
