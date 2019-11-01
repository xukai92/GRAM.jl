using DensityRatioEstimation: DensityRatioEstimation, gaussian_gram_by_pairwise_sqd, pairwise_sqd, MMDAnalytical, _estimate_ratio

function DensityRatioEstimation.adddiag(mmd::MMDAnalytical{T, Val{:true}}, Kdede::Union{MLToolkit.Flux.CuArray,MLToolkit.Tracker.TrackedArray}) where {T}
    ϵ = diagm(0 => mmd.ϵ * fill(one(T), size(Kdede, 1)))
    return Kdede + gpu(ϵ)
end

_compute_mmd_sq(Kdede, Kdenu, Knunu) = mean(Kdede) - 2mean(Kdenu) + mean(Knunu)

function compute_mmd(x_de, x_nu; σs=[], verbose=false)
    function f(pdot_dede, pdot_denu, pdot_nunu, σ)
        Kdede = gaussian_gram_by_pairwise_sqd(pdot_dede, σ)
        Kdenu = gaussian_gram_by_pairwise_sqd(pdot_denu, σ)
        Knunu = gaussian_gram_by_pairwise_sqd(pdot_nunu, σ)
        return _compute_mmd_sq(Kdede, Kdenu, Knunu)
    end
    mmd_sq = multi_run(f, x_de, x_nu, σs, verbose)
    return sqrt(mmd_sq + 1f-6)
end

function estimate_ratio(x_de, x_nu; σs=[], verbose=false)
    function f(pdot_dede, pdot_denu, pdot_nunu, σ)
        Kdede = gaussian_gram_by_pairwise_sqd(pdot_dede, σ)
        Kdenu = gaussian_gram_by_pairwise_sqd(pdot_denu, σ)
        return _estimate_ratio(MMDAnalytical(1f-3), Kdede, Kdenu)
    end
    return multi_run(f, x_de, x_nu, σs, verbose) / convert(Float32, length(σs))
end

function estimate_ratio_compute_mmd(x_de, x_nu; σs=[], verbose=false)
    function f(pdot_dede, pdot_denu, pdot_nunu, σ)
        Kdede = gaussian_gram_by_pairwise_sqd(pdot_dede, σ)
        Kdenu = gaussian_gram_by_pairwise_sqd(pdot_denu, σ)
        Knunu = gaussian_gram_by_pairwise_sqd(pdot_nunu, σ)
        return (_estimate_ratio(MMDAnalytical(1f-3), Kdede, Kdenu), _compute_mmd_sq(Kdede, Kdenu, Knunu))
    end
    ratio, mmd_sq = multi_run(f, x_de, x_nu, σs, verbose)
    return (
        ratio=ratio / convert(Float32, length(σs)), 
        mmd=sqrt(mmd_sq + 1f-6)
    )
end

function multi_run_verbose(σ, verbose)
    if verbose
        @info "Automatically choose σ using the median of pairwise distances: $σ."
    end
    @tb @info "train" σ_median=σ log_step_increment=0
end

Flux.Zygote.@nograd multi_run_verbose

function multi_run(f_run, x_de, x_nu, σs, verbose)
    pdot_dede = pairwise_sqd(x_de)
    pdot_denu = pairwise_sqd(x_de, x_nu)
    pdot_nunu = pairwise_sqd(x_nu)
    
    if isempty(σs)
        σ = sqrt(median(vcat(vec.(Tracker.data.([pdot_dede, pdot_denu, pdot_nunu])))))
        multi_run_verbose(σ, verbose)
        σs = [σ]
    end
    
    res = f_run(pdot_dede, pdot_denu, pdot_nunu, σs[1])
    for σ in σs[2:end]
        _res = f_run(pdot_dede, pdot_denu, pdot_nunu, σ)
        if res isa Tuple
            res = tuple((res[i] .+ _res[i] for i in 1:length(res))...)
        else
            res += _res
        end
    end

    return res
end

###

abstract type AbstractGenerativeModel <: Trainable end

# MMDNet

struct MMDNet <: AbstractGenerativeModel
    logger
    opt
    σs
    g::NeuralSampler
end

Flux.functor(m::MMDNet) = (m.g,), t -> MMDNet(m.logger, m.opt,  m.σs, t[1])

function MLToolkit.Neural.loss(m::MMDNet, x_data)
    x_gen = rand(m.g)
    mmd = compute_mmd(x_gen, x_data; σs=m.σs)
    loss_g = mmd
    return (loss_g=loss_g, mmd=mmd,)
end

function evaluate_g(g, dataset)
    fig = plt.figure(figsize=(3.5, 3.5))
    plot!(dataset, g)
    return (gen=fig,)
end

evaluate(m::MMDNet, dl) = evaluate_g(m.g, dl.dataset)

# GAN

struct GAN <: AbstractGenerativeModel
    logger
    opt
    g::NeuralSampler
    d::Discriminator
end

Flux.functor(m::GAN) = (m.g, m.d), t -> GAN(m.logger, m.opt, t[1], t[2])

BCE = Flux.binarycrossentropy

function update!(opt, m::GAN, x_data)
    y_real, y_fake = 1, 0
    
    # Update d
    ps_d = Flux.params(m.d)
    local accuracy_d, loss_d
    gs_d = gradient(ps_d) do
        x_gen = rand(m.g)
        n_real = last(size(x_data))
        ŷ_d_all = m.d(hcat(x_data, x_gen))
        ŷ_d_real, ŷ_d_fake = ŷ_d_all[:,1:n_real], ŷ_d_all[:,n_real+1:end]
        accuracy_d = (sum(ŷ_d_real .> 0.5f0) + sum(ŷ_d_fake .< 0.5f0)) / length(ŷ_d_all)
        loss_d = (sum(BCE.(ŷ_d_real, y_real)) + sum(BCE.(ŷ_d_fake, y_fake))) / length(ŷ_d_all)
    end
    Flux.Optimise.update!(opt, ps_d, gs_d)

    # Update g
    ps_g = Flux.params(m.g)
    local accuracy_g, loss_g
    gs_g = gradient(ps_g) do
        x_gen = rand(m.g)
        ŷ_g_fake = m.d(x_gen)
        accuracy_g = mean(ŷ_g_fake .< 0.5f0)
        loss_g = mean(BCE.(ŷ_g_fake, y_real))
    end
    Flux.Optimise.update!(opt, ps_g, gs_g)
    
    return (
        loss_d=loss_d,
        loss_g=loss_g, 
        accuracy_d=accuracy_d,
        accuracy_g=accuracy_g
    )
end

evaluate(m::GAN, dl) = evaluate_g(m.g, dl.dataset)

# RMMMDNet

const istraining = Ref(:false)
Flux.istraining() = istraining[]

struct RMMMDNet <: AbstractGenerativeModel
    logger
    opt
    σs
    g::NeuralSampler
    f::Projector
end

Flux.functor(m::RMMMDNet) = (m.g, m.f), t -> RMMMDNet(m.logger, m.opt, m.σs, t[1], t[2])

function update!(opt, m::RMMMDNet, x_data)
    # Training mode
    istraining[] = true
    ps_g = trackerparams(m.g)
    ps_f = trackerparams(m.f)

    # Forward
    x_gen = rand(m.g)
    fx_gen, fx_data = m.f(x_gen), m.f(x_data)
    ratio, mmd = estimate_ratio_compute_mmd(fx_gen, fx_data; σs=m.σs)
    pearson_divergence = mean((ratio .- 1) .^ 2)
    raito_mean = mean(ratio)
    loss_f = -(pearson_divergence + raito_mean)
    loss_g = mmd

    # Collect gradients
    gs_f = gradient(() -> loss_f, ps_f; once=false)
    Tracker.zero_grad!.(Tracker.grad.(ps_g)) # gradient call above leads to un-zeroed gradients in `ps_g`
    
    gs_g = gradient(() -> loss_g, ps_g)
    Tracker.zero_grad!.(Tracker.grad.(ps_f)) # gradient call above leads to un-zeroed gradients in `ps_f`
    
    # Update f and g
    Flux.Optimise.update!(opt, ps_f, gs_f)
    Flux.Optimise.update!(opt, ps_g, gs_g)

    ratio_orig = estimate_ratio(x_gen |> Tracker.data, x_data; σs=m.σs)
    return (
        squared_distance=mean((ratio_orig - Tracker.data(ratio)) .^ 2),
        pearson_divergence=pearson_divergence,
        raito_mean=raito_mean,
        loss_f=loss_f,
        mmd=mmd,
        loss_g=loss_g,
    )
end

function evaluate(m::RMMMDNet, dl)
    istraining[] = false
    fig_g = plt.figure(figsize=(3.5, 3.5))
    plot!(dl.dataset, m.g)
    if m.f.Dout != 2    # only visualise projector if its output dim is 2
        return (gen=fig_g,)
    end
    fig_f = plt.figure(figsize=(3.5, 3.5))
    plot!(dl.dataset, m.g, m.f)
    return (gen=fig_g, proj=fig_f)
end
