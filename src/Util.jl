using LinearAlgebra
using Random
using Distributions
# using Plots



@inline distance(a, b) = norm(a .- b)

# distance(a, b) = sqrt(sum(abs2, a .- b) + 1e-12)


function pairwise_difference_matrix(n::Int)
    pairs = [(i, j) for i in 1:n-1 for j in i+1:n]
    return [i == p[1] ? 1 : i == p[2] ? -1 : 0 for p in pairs, i in 1:n]
end

num_pairwise_differences(n::Int) = n * (n - 1) ÷ 2


function synthetic_tdoa(positions, receivers, c, σ; seed=nothing) 
    
    if seed !== nothing
        Random.seed!(seed)
    end

    N = length(receivers)

    Q = pairwise_difference_matrix(N)

    QN = num_pairwise_differences(N)

    ε = Normal(0, σ)

    map(positions) do source
        delays = [distance(source, receiver) / c for receiver ∈ receivers] 

        return Q * delays + rand(ε, QN)
    end 
end 


function synthetic_toa(positions, event_times, receivers, c, σ; seed=nothing)
    @assert(length(positions) == length(event_times))

    if seed !== nothing
        Random.seed!(seed)
    end

    N = length(receivers)

    ε = Normal(0, σ)

    map(zip(positions, event_times)) do (source, time)
        delays = [distance(source, receiver) / c for receiver ∈ receivers] 

        time .+ delays + rand(ε, N) 
    end 
end 




function stochastic_trajectory(amax; d = 3, tmax=1.0, Δt=0.01,
                               x0=zeros(d), v0=zeros(d), seed=nothing)

    if seed !== nothing
        Random.seed!(seed)
    end

    steps = Int(round(tmax/Δt))
    times = range(0, step=Δt, length=steps+1)

    x = x0
    v = v0

    positions = Vector{Vector{Float64}}(undef, steps+1)

    positions[1] = x

    for i in 1:steps
        # Random acceleration direction
        dir = rand(Uniform(-1, 1), d)
        dir /= norm(dir)

        a = rand(Uniform(0, amax))

        acc = a * dir
        
        # Semi-implicit Euler method
        v += acc * Δt
        x += v * Δt

        positions[i+1] = x
    end

    return times, positions
end


# function rotate(p)
#     anim = Animation()
#     plot!(p, legend=false)
#     for i in range(0, stop = 359, step = 10)
#         fram = Plots.plot(p, camera=(i, 10))
#         frame(anim, fram)
#     end
#     gif(anim, fps=10)
# end 


using KernelDensity
using Optim

function kde_mode(samples) 
    kde_est = kde(samples)
    f(x) = -pdf(kde_est, x[1]) 
    result = optimize(f, [mean(samples)])
    return Optim.minimizer(result)[1]
end 
