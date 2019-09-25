gaussian_gram_by_pairwise_sqd(pdot, σ) = exp.(-pdot ./ 2(σ ^ 2))

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

function _estimate_ratio(Kdede, Kdenu)
    n_de, n_nu = size(Kdenu)
    eps = diagm(0 => fill(1f-3 * one(Float32), size(Kdede, 1)))
    eps = use_gpu.x ? gpu(eps) : eps
    Kdede_stable = Kdede + eps
    return convert(Float32, n_de / n_nu) * (Kdede_stable \ sum(Kdenu; dims=2)[:,1])
end

function estimate_ratio(x_de, x_nu; σs=[], verbose=false)
    function f(pdot_dede, pdot_denu, pdot_nunu, σ)
        Kdede = gaussian_gram_by_pairwise_sqd(pdot_dede, σ)
        Kdenu = gaussian_gram_by_pairwise_sqd(pdot_denu, σ)
        return _estimate_ratio(Kdede, Kdenu)
    end
    return multi_run(f, x_de, x_nu, σs, verbose) / convert(Float32, length(σs))
end

function estimate_ratio_compute_mmd(x_de, x_nu; σs=[], verbose=false)
    function f(pdot_dede, pdot_denu, pdot_nunu, σ)
        Kdede = gaussian_gram_by_pairwise_sqd(pdot_dede, σ)
        Kdenu = gaussian_gram_by_pairwise_sqd(pdot_denu, σ)
        Knunu = gaussian_gram_by_pairwise_sqd(pdot_nunu, σ)
        return (_estimate_ratio(Kdede, Kdenu), _compute_mmd_sq(Kdede, Kdenu, Knunu))
    end
    ratio, mmd_sq = multi_run(f, x_de, x_nu, σs, verbose)
    return (
        ratio=ratio / convert(Float32, length(σs)), 
        mmd=sqrt(mmd_sq + 1f-6)
    )
end

# TODO: implement running average of median
function multi_run(f_run, x_de, x_nu, σs, verbose)
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

    return mapreduce(
        σ -> f_run(pdot_dede, pdot_denu, pdot_nunu, σ), 
        (x, y) -> x .+ y, 
        σs
    )
end

###

abstract type AbstractGenerativeModel end

function update_by_loss!(loss, ps, opt; n=1)
    gs = Tracker.gradient(() -> loss, ps)
    for i in 1:n
        Tracker.update!(opt, ps, gs)
    end
end

using Tracker: Params, losscheck, @interrupts, Grads, tracker, extract_grad!, zero_grad!, grad

function Tracker.gradient(f, xs::Params; once=true)
  l = f()
  losscheck(l)
  @interrupts back!(l; once=once)
  gs = Grads()
  for x in xs
    gs[tracker(x)] = extract_grad!(x)
  end
  return gs
end

function train!(m::AbstractGenerativeModel, n_epochs::Int, dl::DataLoader)
    @showprogress for epoch in 1:n_epochs
        with_logger(m.logger) do
            for (x_data,) in dl.train
                # Step training
                step_info = step!(m, x_data)
                m.iter.x += 1
                # Logging
                Flux.testmode!(m)
                @info "train" step_info...
                if m.iter.x % 10 == 0
                    @info "eval" evaluate(dl.data, m)... log_step_increment=0
                end
            end
        end
        # Save model after each full pass of the dataset
        save!(m)
    end
end

# GAN

struct GAN <: AbstractGenerativeModel
    iter::Base.RefValue{Int}
    logger::AbstractLogger
    g::Generator
    ps_g
    d::Discriminator
    ps_d
    opt
end

Flux.mapchildren(f, m::GAN) = GAN(m.iter, m.logger, f(m.g), m.ps_g, f(m.d), m.ps_d, m.opt)
Flux.children(m::GAN) = (m.iter, m.logger, m.g, m.ps_g, m.d, m.ps_d, m.opt)

function step!(m::GAN, x_data)
    # Training mode
    Flux.testmode!(m.d, false)
    Flux.testmode!(m.g, false)
    
    # Update d
    zero_grad!.(grad.(m.ps_g))
    zero_grad!.(grad.(m.ps_d))
    x_gen = rand(m.g)
    logitŷ_d = m.d(hcat(x_data, x_gen))
    batch_size, batch_size_gen = last(size(x_data)), last(size(x_gen))
    y_real, y_fake = ones(1, batch_size), zeros(1, batch_size_gen)
    y_d = hcat(y_real, y_fake)
    loss_d = mean(Flux.logitbinarycrossentropy.(logitŷ_d, y_d))
    update_by_loss!(loss_d, m.ps_d, m.opt)

    # Update g
    zero_grad!.(grad.(m.ps_g))
    zero_grad!.(grad.(m.ps_d))
    x_gen = rand(m.g)
    logitŷ_g = m.d(x_gen)
    y_g = y_real
    loss_g = mean(Flux.logitbinarycrossentropy.(logitŷ_g, y_g))
    update_by_loss!(loss_g, m.ps_g, m.opt)

    return (
        loss_d=loss_d,
        loss_g=loss_g, 
        accuracy=mean(float.(Flux.data(logitŷ_d) .> 0) .== y_d),
    )
end

function evaluate(d::Data, m::GAN)
    fig = plt.figure(figsize=(3.5, 3.5))
    plot!(d, m.g)
    return (gen=fig,)
end

# MMDNet

struct MMDNet <: AbstractGenerativeModel
    iter::Base.RefValue{Int}
    logger::AbstractLogger
    g::Generator
    ps_g
    opt
    σs
end

Flux.mapchildren(f, m::MMDNet) = MMDNet(m.iter, m.logger, f(m.g), m.ps_g, m.opt, m.σs)
Flux.children(m::MMDNet) = (m.iter, m.logger, m.g, m.ps_g, m.opt, m.σs)

function step!(m::MMDNet, x_data)
    # Training mode
    Flux.testmode!(m, false)

    # Forward
    x_gen = rand(m.g)
    mmd = compute_mmd(x_gen, x_data; σs=m.σs)
    loss_g = mmd
    update_by_loss!(loss_g, m.ps_g, m.opt)

    return (mmd=mmd, loss_g=loss_g,)
end

function evaluate(d::Data, m::MMDNet)
    fig = plt.figure(figsize=(3.5, 3.5))
    plot!(d, m.g)
    return (gen=fig,)
end

# RMMMDNet

struct RMMMDNet <: AbstractGenerativeModel
    iter::Base.RefValue{Int}
    logger::AbstractLogger
    g::Generator
    ps_g
    f::Projector
    ps_f
    opt
    σs
end

function Flux.mapchildren(f, m::RMMMDNet)
    return RMMMDNet(m.iter, m.logger, f(m.g), m.ps_g, f(m.f), m.ps_f, m.opt, m.σs)
end
Flux.children(m::RMMMDNet) = (m.iter, m.logger, m.g, m.ps_g, m.f, m.ps_f, m.opt, m.σs)

function step!(m::RMMMDNet, x_data)
    # Training mode
    Flux.testmode!(m.f, false)
    Flux.testmode!(m.g, false)

    # Forward
    x_gen = rand(m.g)
    fx_gen, fx_data = m.f(x_gen), m.f(x_data)
    ratio, mmd = RMMMDNets.estimate_ratio_compute_mmd(fx_gen, fx_data; σs=m.σs)
    pearson_divergence = mean((ratio .- 1) .^ 2)
    raito_mean = mean(ratio)
    # loss_f_multiplier = 1e-1 / size(fx_gen, 1)    # experimental; not used
    loss_f = -(pearson_divergence + raito_mean)
    loss_g = mmd

    # Collect gradients
    zero_grad!.(grad.(m.ps_g))
    zero_grad!.(grad.(m.ps_f))
    gs_f = Tracker.gradient(() -> loss_f, m.ps_f; once=false)

    zero_grad!.(grad.(m.ps_g))
    zero_grad!.(grad.(m.ps_f))
    gs_g = Tracker.gradient(() -> loss_g, m.ps_g)
    
    # Update f and g
    Tracker.update!(m.opt, m.ps_f, gs_f)
    Tracker.update!(m.opt, m.ps_g, gs_g)

    ratio_orig = estimate_ratio(x_gen |> Flux.data, x_data; σs=m.σs) |> Flux.data
    return (
        squared_distance=mean((ratio_orig - Flux.data(ratio)) .^ 2),
        pearson_divergence=pearson_divergence,
        raito_mean=raito_mean,
        loss_f=loss_f,
        mmd=mmd,
        loss_g=loss_g,
    )
end

function evaluate(d::Data, m::RMMMDNet)
    fig_g = plt.figure(figsize=(3.5, 3.5))
    plot!(d, m.g)
    if m.f.D_fx != 2    # only visualise projector if D_fx is 2
        return (gen=fig_g,)
    end
    fig_f = plt.figure(figsize=(3.5, 3.5))
    plot!(d, m.g, m.f)
    return (gen=fig_g, proj=fig_f)
end
