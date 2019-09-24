module RMMMDNets

const use_gpu = Ref(false)
enable_gpu() = (use_gpu.x = true)
disable_gpu() = (use_gpu.x = false)
export enable_gpu, disable_gpu

using Statistics, LinearAlgebra, StatsFuns, Distributions
using Logging, TensorBoardLogger, Humanize, Dates
using Flux, Flux.Data.MNIST, Tracker, CuArrays
using ProgressMeter: @showprogress
using Random: shuffle, MersenneTwister, AbstractRNG, GLOBAL_RNG
include("anonymized.jl")

###

function parse_toml(toml, dataset, model_name)
    # Extract from TOML dict
    args_dict_strkey = merge(toml["common"], filter(p -> !(p.second isa Dict), toml[dataset]), toml[dataset][model_name])
    # Convert keys to Symbol
    args_dict = Dict(Symbol(p.first) => p.second for p in args_dict_strkey)
    # Add dataset and model_name
    args_dict[:dataset] = dataset
    args_dict[:model_name] = model_name
    return args_dict
end

function parse_args_dict(_args_dict; override::NamedTuple=NamedTuple(), suffix::String="")
    # Imutability
    args_dict = copy(_args_dict)
    # Oeverride
    for k in keys(override)
        @assert k in keys(args_dict) "Cannot overrid unexistent keys: $k"
        args_dict[k] = override[k]
    end
    # Show arguments
    @info "Args" args_dict...
    # Generate experiment name from dict
    exclude = [
        :seed,
        :dataset,
        :model_name,
        :n_epochs,
        :act_last,
    ]
    exp_name = flatten_dict(args_dict; exclude=exclude)
    exp_name *= "-seed=$(args_dict[:seed])" # always put seed in the end 
    if suffix != ""
        exp_name *= "-$suffix"
    end
    # Add exp_name to args_dict
    args_dict[:exp_name] = exp_name
    # Parse "1,2,3" => [1,2,3]
    parse_csv(T, l) = map(x -> parse(T, x), split(l, ","))
    if :sigma in keys(args_dict)
        args_dict[:sigma] = if args_dict[:sigma] == "median"; []
        else parse_csv(Float32, args_dict[:sigma]) end
    end
    args_dict[:Dg_h] = parse_csv(Int, args_dict[:Dg_h])
    if :Dd_h in keys(args_dict)
        args_dict[:Dd_h] = parse_csv(Int, args_dict[:Dd_h])
    end
    if :Df_h in keys(args_dict)
        args_dict[:Df_h] = parse_csv(Int, args_dict[:Df_h])
    end
    # Convert dict to named tuple
    args = dict2namedtuple(args_dict)
    return args
end

export parse_toml, parse_args_dict

###

include("data.jl")
export Data, DataLoader
include("modules.jl")
export Generator, Projector
include("models.jl")
export GAN, MMDNet, RMMMDNet, train!, evaluate

###

function get_data(dataset::String)
    rng = MersenneTwister(1)
    if dataset == "mnist"
        X_train = Array{Float32,2}(hcat(float.(reshape.(MNIST.images(), :))...))
        dim = size(X_train, 1)
    end
    if dataset == "gaussian"
        N = 1_000
        X_train = rand(rng, MvNormal(zeros(Float32, 2), Float32[0.25 0.25; 0.25 1]), N)
    end
    if dataset == "ring"
        N = 1_000
        X_train = rand(rng, Ring(8, 2, 2f-1), N)
    end
    X_train = use_gpu.x ? gpu(X_train) : X_train
    return Data(dataset, X_train)
end

function plot!(d::Data, g::Generator)
    rng = MersenneTwister(1)
    if d.dataset == "mnist" 
        x_data = d.train
        x_gen = rand(rng, g, 32) |> cpu |> Flux.data
        plot_grayimg!(hcat(x_data[:,shuffle(rng, 1:last(size(x_data)))[1:32]], x_gen))
    elseif d.dataset in ["gaussian", "ring"]
        x_data = d.train
        x_gen = rand(rng, g, last(size(x_data))) |> cpu |> Flux.data
        plt.scatter(x_data[1,:], x_data[2,:], marker=".", label="data", alpha=0.5)
        plt.scatter(x_gen[1,:],  x_gen[2,:],  marker=".", label="gen",  alpha=0.5)
        autoset_lim!(x_data)
        plt.legend(fancybox=true, framealpha=0.5)
    end
end

function plot!(d::Data, g::Generator, f::Projector)
    rng = MersenneTwister(1)
    if d.dataset == "mnist" 
        x_data = d.train
        x_gen = rand(rng, g, 32) 
        fx_data = f(x_data) |> cpu |> Flux.data
        fx_gen = f(x_gen) |> cpu |> Flux.data
        plot_grayimg!(hcat(fx_data[:,shuffle(rng, 1:last(size(fx_data)))[1:32]], fx_gen))
    elseif d.dataset in ["gaussian", "ring"]
        x_data = d.train
        x_gen = rand(rng, g, last(size(x_data)))
        fx_data = f(x_data) |> cpu |> Flux.data
        fx_gen = f(x_gen) |> cpu |> Flux.data
        plt.scatter(fx_data[1,:], fx_data[2,:], marker=".", label="data", alpha=0.5)
        plt.scatter(fx_gen[1,:],  fx_gen[2,:],  marker=".", label="gen",  alpha=0.5)
        autoset_lim!(fx_data)
        plt.legend(fancybox=true, framealpha=0.5)
    end
end

function get_model(args::NamedTuple, data::Data)
    module_path = pathof(@__MODULE__) |> splitdir |> first |> splitdir |> first
    logdir = "$(data.dataset)/$(args.model_name)/$(args.exp_name)/$(Dates.format(now(), DATETIME_FMT))"
    logger = TBLogger("$module_path/logs/$logdir")
    if args.opt == "adam"
        opt = ADAM(args.lr, (args.beta1, 999f-4))
    elseif args.opt == "rmsprop"
        opt = RMSProp(args.lr)
    end
    if args.base == "uniform"
        base = UniformBase(args.D_z)
    elseif args.base == "gaussian"
        base = GaussianBase(args.D_z)
    end
    g = Generator(base, args.D_z, args.Dg_h, data.dim, args.act, args.act_last, args.norm, args.batch_size_gen)
    if args.model_name == "gan"
        d = Discriminator(data.dim, args.Dd_h, args.act, args.norm)
        m = GAN(Ref(0), logger, g, Flux.params(g), d, Flux.params(d), opt)
    elseif args.model_name == "mmdnet"
        m = MMDNet(Ref(0), logger, g, Flux.params(g), opt, args.sigma)
    elseif args.model_name == "rmmmdnet"
        f = Projector(data.dim, args.Df_h, args.D_fx, args.act, args.norm)
        m = RMMMDNet(Ref(0), logger, g, Flux.params(g), f, Flux.params(f), opt, args.sigma)
    end
    @info "Init $(args.model_name) with $(nparams(m) |> Humanize.digitsep) parameters" logdir
    m = use_gpu.x ? gpu(m) : m
    return m
end

export get_data, get_model

###

using BSON

function save!(model::AbstractGenerativeModel)
    model_cpu = model |> RMMMDNets.Flux.cpu
    model_fname = "$(model.logger.logdir)/model.bson"
    iter = model.iter
    weights = Tracker.data.(Flux.params(model))
    bson(model_fname, Dict(:iter => model.iter, :weights => weights))
    @info "Saved model at $(iter.x) iterations to $model_fname"
    return model_fname
end

function load!(model::AbstractGenerativeModel, model_fname::String)
    model_loaded = BSON.load(model_fname)
    weights = model_loaded[:weights]
    Flux.loadparams!(model, weights)
    iter = model_loaded[:iter]
    model.iter.x = iter.x
    @info "Loaded model at $(iter.x) iterations from $model_fname"
end

export save!, load!

end # module
