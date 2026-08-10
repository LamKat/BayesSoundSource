using Statistics


function zscore(x) 
    @. (x - mean(x)) / std(x) 
end 


function outlier_posterior_assignment(dist::MixtureModel, y)
    logw    = log.(probs(dist)) .+ logpdf.(components(dist), y)
    # return softmax(logw)
    # Gets the posterior assignment of the second mixture model component
    return logistic(logw[2] - logw[1])
end

@model function outlier_mixture(x, y)
    N = length(x)
    ℓ ~ LogNormal(0.0, 1.0)
    σ ~ Exponential(0.5)
    
    kernel = with_lengthscale(SqExponentialKernel(), ℓ)

    K = kernelmatrix(kernel, x) + 1e-5*I     
    f ~ MvNormal(zeros(N), K)

    π ~ Uniform(0, 1)

    resp = Vector(undef, N)
    
    for i ∈ 1:N 
        dist = MixtureModel([Normal(f[i], σ), Cauchy()], [π, 1-π])
        y[i] ~ dist 
        resp[i] = outlier_posterior_assignment(dist, y[i])
    end 

    return resp

end


