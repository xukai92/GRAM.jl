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

using DensityRatioEstimation: MMDAnalytical, _estimate_ratio, pairwise_sqd, gaussian_gram_by_pairwise_sqd

function estimate_ratio_and_compute_mmd_sq(pdot_dede, pdot_denu, pdot_nunu, σ)
    Kdede = gaussian_gram_by_pairwise_sqd(pdot_dede, σ)
    Kdenu = gaussian_gram_by_pairwise_sqd(pdot_denu, σ)
    Knunu = gaussian_gram_by_pairwise_sqd(pdot_nunu, σ)
    ratio = _estimate_ratio(MMDAnalytical(), Kdede, Kdenu)
    mmd_sq = mean(Kdede) - 2mean(Kdenu) + mean(Knunu)
    return ratio, mmd_sq
end

# TODO: implement running average of median
function estimate_ratio_and_compute_mmd(x_de, x_nu; σs=[], verbose=false)
    pdot_dede = pairwise_sqd(x_de)
    pdot_denu = pairwise_sqd(x_de, x_nu)
    pdot_nunu = pairwise_sqd(x_nu)
    
    if isempty(σs)
        σ = sqrt(median(vcat(vec.(Flux.data.([pdot_dede, pdot_denu, pdot_nunu])))))
        if verbose
            @info "Automatically choose σ using the median of pairwise distances: $σ."
        end
        @tb @info "train" σ_median=σ log_step_increment=0
        σs = [σ]
    end

    ratio, mmd_sq = mapreduce(
        σ -> estimate_ratio_and_compute_mmd_sq(pdot_dede, pdot_denu, pdot_nunu, σ), 
        (t1, t2) -> (t1[1] + t2[1], t1[2] + t2[2]), 
        σs
    )
    
    n = convert(Float32, length(σs))
    return (ratio=ratio / n, mmd=sqrt(mmd_sq + 1f-6))
end

###

function get_data(dataset::String)
    rng = MersenneTwister(1234)
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
        X_train = rand(rng, Ring(8, 1, 1f-1), N)
    end
    X_train = use_gpu.x ? gpu(X_train) : X_train
    return Data(dataset, X_train)
end

function vis(d::Data, g::Generator)
    rng = MersenneTwister(1234)
    fig = plt.figure(figsize=(3.5, 3.5))
    if d.dataset == "mnist" 
        X_data = d.train
        X_gen = rand(rng, g, 32) |> cpu |> Flux.data
        plot_grayimg!(hcat(X_data[:,shuffle(rng, 1:last(size(X_data)))[1:32]], X_gen))
    end
    if d.dataset in ["gaussian", "ring"]
        X_data = d.train
        X_gen = rand(rng, g, 200) |> cpu |> Flux.data
        plt.scatter(X_data[1,:], X_data[2,:], marker=".", label="data", alpha=0.5)
        plt.scatter(X_gen[1,:],  X_gen[2,:],  marker=".", label="gen",  alpha=0.5)
        autoset_lim!(X_data)
        plt.legend(fancybox=true, framealpha=0.5)
    end
    return fig
end

function get_model(model_name::String, args::NamedTuple, data::Data, exp_name::String)
    module_path = pathof(@__MODULE__) |> splitdir |> first |> splitdir |> first
    logdir = "$(data.dataset)/$model_name/$exp_name/$(Dates.format(now(), DATETIME_FMT))"
    logger = TBLogger("$module_path/logs/$logdir")
    if args.opt == "adam_akash" # Akash's secrete setting
        opt = ADAM(1f-4, (5f-1, 999f-4))
        @warn "`args.lr` is ignored for `adam_akash`"
    end
    if args.opt == "adam"
        opt = ADAM(args.lr)
    end
    if args.opt == "rmsprop"
        opt = RMSProp(args.lr)
    end
    g = Generator(args.D_z, args.D_h, data.dim, args.σ, args.σ_last)
    if model_name == "mmdnet"
        m = MMDNet(Ref(0), logger, g, opt, args.σs)
    end
    if model_name == "rmmmdnet"
        f = Projector(data.dim, args.D_h, args.D_fx, args.σ)
        m = RMMMDNet(Ref(0), logger, g, f, opt, args.σs)
    end
    @info "Init $model_name with $(nparams(m) |> Humanize.digitsep) parameters" logdir
    m = use_gpu.x ? gpu(m) : m
    return m
end

export get_data, vis, get_model

end # module
