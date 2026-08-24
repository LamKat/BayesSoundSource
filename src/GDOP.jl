
using LinearAlgebra

## Based on Regina Kaune, "Accuracy Studies for TDOA and TOA Localization"
## http://fusion.isif.org/proceedings/fusion12CD/html/pdf/056_271.pdf


"""
    crlb_toa(receivers, source; speed_of_sound=1.0, σ=1.0)

Compute the Cramér–Rao lower bound (CRLB) for source localisation from
time-of-arrival (TOA) measurements.

Assumes independent, identically distributed Gaussian TOA errors with
standard deviation `σ`. 

# Returns
A `(dims + 1) × (dims + 1)` matrix containing the CRLB for the spatial
coordinates and emission time.
"""
function crlb_toa(receivers, source; speed_of_sound=1.0, σ=1.0) 
    @assert source ∉ receivers "This method breaks down when computing crlb at a receiver"
    @assert allequal(length(receivers))

    dims = length(receivers[1])
    num_sensors = length(receivers)
    los = [source .- receiver for receiver in receivers]  ## line-of-sight 
    los = los ./ norm.(los)

    H = Matrix{Float64}(undef, num_sensors, dims + 1)
    for k ∈ eachindex(los)
        # The k-th row of the Jacobian matrix corresponds to TOA_k
        H[k,:] = vcat(los[k] ./ speed_of_sound, 1.0)
    end 

    R = σ^2 * I(num_sensors)

    FIM = transpose(H) * inv(R)  * H  
    return pinv(FIM) ## pseudoinverse
end 


"""
    crlb_tdoa(receivers, source; speed_of_sound=1.0, σ=1.0)

Compute the Cramér–Rao lower bound (CRLB) for source localisation from
time-difference-of-arrival (TDOA) measurements.

All receiver pairs are treated as measurements with independent, identically
distributed Gaussian errors with standard deviation `σ`. 

# Returns
A `dims × dims` matrix containing the CRLB for the source coordinates.
"""
function crlb_tdoa(receivers, source; speed_of_sound=1.0, σ=1.0)
    @assert source ∉ receivers "This method breaks down when computing crlb at a receiver"
    @assert allequal(length(receivers))

    dims = length(receivers[1])
    num_sensors = length(receivers)
    num_pairs = num_sensors * (num_sensors - 1) ÷ 2

    pairs = [(i, j) for i ∈ 1:num_sensors for j ∈ i+1:num_sensors]

    # Calculate the direction from each receiver on the unit sphere
    los = [source .- receiver for receiver in receivers]
    los = los ./ norm.(los)

    H = Matrix{Float64}(undef, num_pairs, dims)
    for (k, (i,j)) ∈ enumerate(pairs)
        # The k-th row of the Jacobian matrix corresponds to TDOA_ij
        H[k,:] = (los[i] - los[j]) ./ speed_of_sound
    end 

    R = σ^2 * I(num_pairs)

    # Compute the Fisher Information Matrix (FIM)
    FIM = H' * inv(R) * H

    return pinv(FIM)
end 

"""
    gdop_toa(receivers, source)

Compute a scalar geometric dilution of precision (GDOP) from
time-of-arrival (TOA) measurements.
"""
gdop_toa(receivers, source) = √(tr(crlb_toa(receivers, source)))


"""
    gdop_tdoa(receivers, source)

Compute a scalar geometric dilution of precision (GDOP) from
time-difference-of-arrival (TDOA) measurements.
"""
gdop_tdoa(receivers, source) = √(tr(crlb_tdoa(receivers, source)))
