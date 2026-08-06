using Turing
using MCMCChains
using Distributions
using KernelFunctions
using AbstractGPs
using LinearAlgebra
using KernelDensity


abstract type AbstractTrajPrior end

struct GPTrajPrior   <: AbstractTrajPrior 
    dim::Int
end
(prior::GPTrajPrior)(Z::Int, t=nothing) =
    to_submodel(gp_prior(Z, t, prior))
@model function gp_prior(Z, t, prior::GPTrajPrior) 
    @assert !Base.isnothing(t) "GP trajectory prior requires emission time estimate"
    ℓ ~ Gamma(2,1)
    σ_noise ~ Gamma(2,1)
    σ_rbf ~ Gamma(2,1)

    k = σ_rbf^2 * with_lengthscale(SEKernel(), ℓ)

    f = GP(k)
    f_ = f(t, σ_noise^2)

    coords ~ filldist(f_, prior.dim)

    return coords
end 



struct FlatTrajPrior <: AbstractTrajPrior
    dim::Int
    bounds::Vector{Tuple{Float64,Float64}}

    function FlatTrajPrior(dim::Int,
                           bounds::Vector{Tuple{Float64,Float64}})
        if length(bounds) != dim 
            throw(ArgumentError("expected $dim bounds, got $(length(bounds))"))
        end 
        new(dim, bounds)
    end
end
(prior::FlatTrajPrior)(Z::Int, t=nothing) =
    to_submodel(flat_prior(Z, prior))

@model function flat_prior(Z::Int, prior::FlatTrajPrior)
    # coords = Vector(undef, prior.dim)
    coords = Matrix(undef, prior.dim, Z)

    for d in 1:prior.dim
        lo, hi = prior.bounds[d]
        coords[:, d] ~ filldist(Uniform(lo, hi), Z)
    end

    return coords
end











time_prior(Z::Int) = filldist(Flat(), Z)


struct ConstVector{T}
    value::T
end

Base.getindex(v::ConstVector, i::Int) = v.value


@model function toa_tdoa_model(
    tdoa_obs,
    toa_obs,
    receiver_priors,
    traj_prior;
    Qs = ConstVector(pairwise_difference_matrix(length(receiver_priors))),
    toa_noise_prior = Exponential(0.001),
    tdoa_noise_prior = Exponential(0.001),
    sos_prior = Normal(SPEED_OF_SOUND, 2.0))   
    
    @assert length(tdoa_obs) == length(toa_obs)
    Z = length(toa_obs)

    receivers ~ arraydist(receiver_priors)
    sos ~ sos_prior
    σ_toa ~ toa_noise_prior
    σ_tdoa ~ tdoa_noise_prior
    t ~ time_prior(Z)
    traj ~ traj_prior(Z, t)

    for z in 1:Z
        source = view(traj, z, :)
        delays = [distance(receiver, source) / sos for receiver in eachcol(receivers)]

        tdoa_obs[z] ~ MvNormal(Qs[z] * delays, σ_tdoa)
        toa_obs[z] ~ MvNormal(t[z] .+ delays, σ_toa)
    end
end

@model function tdoa_model(
    tdoa_obs,
    receiver_priors,
    traj_prior;
    Qs = ConstVector(pairwise_difference_matrix(length(receiver_priors))),
    tdoa_noise_prior = Exponential(0.001),
    sos_prior = Normal(SPEED_OF_SOUND, 2.0))  
    
    Z = length(tdoa_obs)

    receivers ~ arraydist(receiver_priors)
    sos ~ sos_prior
    σ_tdoa ~ tdoa_noise_prior

    traj ~ traj_prior(Z, nothing)

    for z in 1:Z
        source = view(traj, z, :)
        delays = [distance(receiver, source) / sos for receiver in eachcol(receivers)]

        tdoa_obs[z] ~ MvNormal(Qs[z] * delays, σ_tdoa)
    end
end

@model function toa_model(
    toa_obs,
    receiver_priors,
    traj_prior;
    toa_noise_prior = Exponential(0.001),
    sos_prior = Normal(SPEED_OF_SOUND, 2.0))  
    
    Z = length(toa_obs)

    receivers ~ arraydist(receiver_priors)
    sos ~ sos_prior
    σ_toa ~ toa_noise_prior
    t ~ time_prior(Z)
    traj ~ traj_prior(Z, t)

    for z in 1:Z
        source = view(traj, z, :)
        delays = [distance(receiver, source) / sos for receiver in eachcol(receivers)]

        toa_obs[z] ~ MvNormal(t[z] .+ delays, σ_toa)
    end
end



function sample_traj(chn, dims, samples=length(chn)) 
    chn_sm = sample(chn, samples)
    Z = traj_length_from_chain(chn, dims)

    m = Array{Float64}(undef, Z, dims, samples)
    for d ∈ 1:dims
        for z ∈ 1:Z
            m[z, d, :] = Array(group(chn_sm, "traj.coords[$z, $d]"))
        end 
    end 
    return m 
end 

traj_length_from_chain(chn, dims) = Z = size(group(chn, "traj.coords"), 2) ÷ dims 

function map_estimate(chn, dims)
    traj = sample_traj(chn, dims)
    map(1:dims) do d
        map(kde_mode, eachrow(traj[:,d,:]))
    end 
end 


function spread(chn, dims)
    Z = traj_length_from_chain(chn, dims)
    traj = sample_traj(chn, dims)
    samples = size(traj, 3)
    map(1:Z) do z 
        z_obs = traj[z,:,:]
        y_map = kde_mode.(eachrow(z_obs))
        return mean(1:samples) do i 
            distance(y_map, z_obs[:, i])
        end 
    end 
end 