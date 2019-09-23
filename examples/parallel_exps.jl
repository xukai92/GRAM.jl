using Distributed
addprocs(4)

using MLToolkit: flatten_dict, dict2namedtuple

@everywhere begin 
    using Random: seed!
    using RMMMDNets
end

###

RMMMDNets_PATH = pathof(RMMMDNets) |> splitdir |> first |> splitdir |> first
include("$RMMMDNets_PATH/scripting.jl")

# dataset = "gaussian"
 dataset = "ring"
# dataset = "mnist"

model_name = "gan"
# model_name = "mmdnet"
# model_name = "rmmmdnet"

args_list = [get_args(
    dataset, 
    model_name; 
    override=(seed=seed, n_epochs=1_000, D_z=20), 
    suffix="seed=$seed"
) for seed in 1:4]

###

@everywhere function run_exp(args)
    data = get_data(args.dataset)
    
    seed!(args.seed)

    model = get_model(args, data)

    evaluate(data, model)

    dataloader = DataLoader(data, args.batch_size)
    
    train!(model, args.n_epochs, dataloader)
end

@sync @distributed for args in args_list
    run_exp(args)
end
