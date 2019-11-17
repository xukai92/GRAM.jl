using DensityRatioEstimation: DensityRatioEstimation, gaussian_gram_by_pairwise_sqd, pairwise_sqd, MMDAnalytical, _estimate_ratio

makecuϵ(T, D, ϵ) = MLToolkit.Flux.CuArray(diagm(0 => ϵ * fill(one(T), D)))

Flux.Zygote.@nograd makecuϵ

DensityRatioEstimation.adddiag(mmd::MMDAnalytical{T,Val{true}}, Kdede) where {T} = 
    Kdede + makecuϵ(T, size(Kdede, 1), mmd.ϵ)

_compute_mmd_sq(Kdede, Kdenu, Knunu) = mean(Kdede) - 2mean(Kdenu) + mean(Knunu)

EPS_RATIO = 1f-3
EPS_MMD = 1f-6

function compute_mmd(x_de, x_nu; σs=[], verbose=false, isclip=false)
    function f(pdot_dede, pdot_denu, pdot_nunu, σ)
        Kdede = gaussian_gram_by_pairwise_sqd(pdot_dede, σ)
        Kdenu = gaussian_gram_by_pairwise_sqd(pdot_denu, σ)
        Knunu = gaussian_gram_by_pairwise_sqd(pdot_nunu, σ)
        return _compute_mmd_sq(Kdede, Kdenu, Knunu)
    end
    mmd_sq = multi_run(f, x_de, x_nu, σs, verbose)
    if isclip
        return sqrt(relu(mmd_sq))
    else
        return sqrt(mmd_sq + EPS_MMD)
    end
end

function estimate_ratio(x_de, x_nu; σs=[], verbose=false)
    function f(pdot_dede, pdot_denu, pdot_nunu, σ)
        Kdede = gaussian_gram_by_pairwise_sqd(pdot_dede, σ)
        Kdenu = gaussian_gram_by_pairwise_sqd(pdot_denu, σ)
        return _estimate_ratio(MMDAnalytical(EPS_RATIO), Kdede, Kdenu)
    end
    return multi_run(f, x_de, x_nu, σs, verbose) / convert(Float32, length(σs))
end

function estimate_ratio_compute_mmd(x_de, x_nu; σs=[], verbose=false)
    function f(pdot_dede, pdot_denu, pdot_nunu, σ)
        Kdede = gaussian_gram_by_pairwise_sqd(pdot_dede, σ)
        Kdenu = gaussian_gram_by_pairwise_sqd(pdot_denu, σ)
        Knunu = gaussian_gram_by_pairwise_sqd(pdot_nunu, σ)
        return (_estimate_ratio(MMDAnalytical(EPS_RATIO), Kdede, Kdenu), _compute_mmd_sq(Kdede, Kdenu, Knunu))
    end
    ratio, mmd_sq = multi_run(f, x_de, x_nu, σs, verbose)
    return (
        ratio=ratio / convert(Float32, length(σs)), 
        mmd=sqrt(mmd_sq + EPS_MMD)
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
        σ = sqrt(median(vcat(vec.([pdot_dede, pdot_denu, pdot_nunu]))))
        multi_run_verbose(σ, verbose)
        σs = [σ]
    end
    
    local res = nothing
    for σ in σs
        res_current = f_run(pdot_dede, pdot_denu, pdot_nunu, σ)
        if isnothing(res)
            res = res_current
        else
            res = res .+ res_current
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
    mmd = compute_mmd(x_gen |> flatten, x_data |> flatten; σs=m.σs)
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

struct RMMMDNet <: AbstractGenerativeModel
    logger
    opt
    σs
    g::NeuralSampler
    f::Projector
end

Flux.functor(m::RMMMDNet) = (m.g, m.f), t -> RMMMDNet(m.logger, m.opt, m.σs, t[1], t[2])

function update!(opt, m::RMMMDNet, x_data)
    # Update f and g
    ps_f, ps_g = Flux.params(m.f), Flux.params(m.g)
    
    local x_gen, ratio, pearson_divergence, raito_mean, mmd, loss_f, loss_g
    gs_f, gs_g = gradient(ps_f, ps_g) do
        x_gen = rand(m.g)
        fx_gen, fx_data = m.f(x_gen), m.f(x_data)
        ratio, mmd = estimate_ratio_compute_mmd(fx_gen, fx_data; σs=m.σs)
        loss_g = mmd
        # Clip version
        ratio = clamp.(ratio, 0f0, 1f9)
        pearson_divergence = mean((ratio .- 1) .^ 2)
        loss_f = -pearson_divergence
        # Regularizer version
#         raito_mean = mean(ratio)
#         pearson_divergence = mean((ratio .- 1) .^ 2)
#         loss_f = -(pearson_divergence + raito_mean)
        (loss_f, loss_g)
    end

    Flux.Optimise.update!(opt, ps_f, gs_f)
    Flux.Optimise.update!(opt, ps_g, gs_g)

    return (
        pearson_divergence=pearson_divergence,
#         raito_mean=raito_mean,
        mmd=mmd,
        loss_f=loss_f,
        loss_g=loss_g,
#         squared_distance=mean((estimate_ratio(x_gen |> flatten, x_data |> flatten; σs=m.σs) - ratio) .^ 2),
    )
end

function evaluate(m::RMMMDNet, dl)
    fig_g = plt.figure(figsize=(3.5, 3.5))
    plot!(dl.dataset, m.g)
    if m.f.Dout != 2    # only visualise projector if its output dim is 2
        return (gen=fig_g,)
    end
    fig_f = plt.figure(figsize=(3.5, 3.5))
    plot!(dl.dataset, m.g, m.f)
    return (gen=fig_g, proj=fig_f)
end

# MMDGAN

struct MMDGAN <: AbstractGenerativeModel
    logger
    opt
    σs
    g::NeuralSampler
    fenc::Projector
    fdec::Projector
end

Flux.functor(m::MMDGAN) = (m.g, m.fenc, m.fdec), t -> MMDGAN(m.logger, m.opt, m.σs, t[1], t[2], t[3])

function forward_and_loss(m::MMDGAN, x_data)
    x_gen = rand(m.g)
    fx_gen, fx_data = m.fenc(x_gen), m.fenc(x_data)
    mmd = compute_mmd(fx_gen, fx_data; σs=m.σs, isclip=true)
    one_side = -mean(relu.(-((mean(fx_gen; dims=2) - mean(fx_data; dims=2)))))
    return mmd, one_side, x_gen, fx_gen, fx_data
end

function update!(opt, m::MMDGAN, x_data)
    ps_fenc, ps_fdec, ps_g = Flux.params(m.fenc), Flux.params(m.fdec), Flux.params(m.g)
    ps_f = Flux.Params([ps_fenc..., ps_fdec...])
    
    # 3d ring fails with λ_ae_data=λ_ae_gen=8f0
    λ_rg, λ_ae_data, λ_ae_gen = 16f0, 8f0, 8f0
    
    # Update f
    local mmd_f, one_side_f, l2_data, l2_gen, loss_f
    for _ in 1:5
        
    # Clamp parameters of f encoder to a cube
    for p in ps_fenc
        p .= clamp.(p, -1f-1, 1f-1)
    end

    gs_f = gradient(ps_f) do
        mmd_f, one_side_f, x_gen, fx_gen, fx_data = forward_and_loss(m, x_data)
        xtilde_gen, xtilde_data = m.fdec(fx_gen), m.fdec(fx_data)
        l2_data = mean((x_data - xtilde_data) .^ 2)
        l2_gen  = mean((x_gen  - xtilde_gen)  .^ 2)
        loss_f = -(mmd_f + λ_rg * one_side_f - λ_ae_data * l2_data - λ_ae_gen * l2_gen)
        loss_f
    end
    Flux.Optimise.update!(opt, ps_f, gs_f)
    
    end
    
    # Update g
    local mmd_g, one_side_g, loss_g
    gs_g = gradient(ps_g) do
        mmd_g, one_side_g, = forward_and_loss(m, x_data)
        loss_g = mmd_g + λ_rg * one_side_g
    end
    Flux.Optimise.update!(opt, ps_g, gs_g)

    return (
        mmd_f=mmd_f,
        one_side_f=one_side_f, 
        l2_data=l2_data, 
        l2_gen=l2_gen,
        loss_f=loss_f,
        mmd_g=mmd_g,
        one_side_g=one_side_g, 
        loss_g=loss_g,
    )
end

function evaluate(m::MMDGAN, dl)
    fig_g = plt.figure(figsize=(3.5, 3.5))
    plot!(dl.dataset, m.g)
    if m.fenc.Dout != 2    # only visualise projector if its output dim is 2
        return (gen=fig_g,)
    end
    fig_f = plt.figure(figsize=(3.5, 3.5))
    plot!(dl.dataset, m.g, m.fenc)
    return (gen=fig_g, proj=fig_f)
end