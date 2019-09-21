module RMMMDNets

const use_gpu = Ref(false)
enable_gpu() = (use_gpu.x = true)
disable_gpu() = (use_gpu.x = false)
export enable_gpu, disable_gpu

using Statistics, LinearAlgebra, StatsFuns, Distributions
using Logging, TensorBoardLogger, Humanize, Dates
using Flux, Flux.Data.MNIST, Tracker
using ProgressMeter: @showprogress
using Random: shuffle, MersenneTwister, AbstractRNG, GLOBAL_RNG
using MLToolkit: DATETIME_FMT, @tb, istb, plt, plot_grayimg!, autoset_lim!, nparams

# @enum Model mmdnet rmmdnet
# @enum Dataset gaussian ring mnist
# export mmdnet, rmmmdnet, gaussian, ring, mnist

include("data.jl")
export Data, DataLoader
include("modules.jl")
export Generator, Projector
include("models.jl")
export MMDNet, RMMMDNet, train!

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

function vis(d::Data, g::Generator)
    rng = MersenneTwister(1)
    fig = plt.figure(figsize=(3.5, 3.5))
    if d.dataset == "mnist" 
        X_data = d.train
        X_gen = rand(rng, g, 32) |> cpu |> Flux.data
        plot_grayimg!(hcat(X_data[:,shuffle(rng, 1:last(size(X_data)))[1:32]], X_gen))
    elseif d.dataset in ["gaussian", "ring"]
        X_data = d.train
        X_gen = rand(rng, g, last(size(X_data))) |> cpu |> Flux.data
        plt.scatter(X_data[1,:], X_data[2,:], marker=".", label="data", alpha=0.5)
        plt.scatter(X_gen[1,:],  X_gen[2,:],  marker=".", label="gen",  alpha=0.5)
        autoset_lim!(X_data)
        plt.legend(fancybox=true, framealpha=0.5)
    end
    return fig
end

function get_model(args::NamedTuple, data::Data)
    module_path = pathof(@__MODULE__) |> splitdir |> first |> splitdir |> first
    logdir = "$(data.dataset)/$(args.model_name)/$(args.exp_name)/$(Dates.format(now(), DATETIME_FMT))"
    logger = TBLogger("$module_path/logs/$logdir")
    if args.opt == "adam_akash" # Akash's secrete setting
        opt = ADAM(args.lr, (5f-1, 999f-4))
    elseif args.opt == "adam"
        opt = ADAM(args.lr)
    elseif args.opt == "rmsprop"
        opt = RMSProp(args.lr)
    end
    if args.base == "uniform"
        base = UniformBase(args.D_z)
    elseif args.base == "gaussian"
        base = GaussianBase(args.D_z)
    end
    g = Generator(base, args.D_z, args.Dg_h, data.dim, args.σ, args.σ_last, args.batch_size_gen)
    if args.model_name == "mmdnet"
        m = MMDNet(Ref(0), logger, g, Flux.params(g), opt, args.σs)
    elseif args.model_name == "rmmmdnet"
        f = Projector(data.dim, args.Df_h, args.D_fx, args.σ)
        m = RMMMDNet(Ref(0), logger, g, Flux.params(g), f, Flux.params(f), opt, args.σs)
    end
    @info "Init $(args.model_name) with $(nparams(m) |> Humanize.digitsep) parameters" logdir
    m = use_gpu.x ? gpu(m) : m
    return m
end

export get_data, vis, get_model

end # module
