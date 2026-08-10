using FFTW
using DSP


PHAT(crosspow) = 1 ./ abs.(crosspow)
bandpass(low, high, freq) = (x) -> low .<= abs.(fftfreq(length(x), freq)) .<= high 
constant(v) = (_) -> v

function pad_zero(s, len::Integer)
    return vcat(s, zeros(len - length(s)))
end

## computes f ⋆ g using the a fft based convolution 
## Returns an array from -len:len
function GCC(f, g; θ = constant(1)) 
    @assert (length(f) == length(g)) "Attempting to convolute signals with differing length"
    signal_length = length(f)

    # pad_length = nextpow(2, signal_length*2)
    pad_length = signal_length*2

    f_fft = pad_zero(f, pad_length) |> fft
    g_fft = pad_zero(g, pad_length) |> fft

    crosspow = g_fft .* conj(f_fft)

    conv = abs.(ifft(crosspow .* θ(crosspow)))

    return fftshift(conv)
end 


delays(len, step) = range(stop=step*(len/2 - 1), length=len, step=step)

function GCC_maximum(f, g, freq, offset, max_delay=Inf, θ=constant(1))
    cc = GCC(f, g; θ=θ)
    λ = 1/freq

    delay_set = delays(length(cc), λ) .- λ*offset

    mask = [-abs(max_delay) .<= delay_set .<= abs(max_delay)]

    # delay = delays(length(cc), λ)[argmax(cc)]


    # return delay - λ*offset

    return (delay_set[mask...])[argmax(cc[mask...])]
end 


iswithin(x, r::AbstractRange) = first(r) <= x <= last(r)

function butterworth_smooth(data, freq, resolution, speed_of_sound)
    fc = speed_of_sound / resolution
    resp = Lowpass(fc) 
    filtfilt(digitalfilter(resp, Butterworth(1); fs=freq),data)
end 

function delay_correlation(f, g, freq, offset; θ=constant(1), smooth=identity)
    cc = GCC(f, g; θ=θ)
    λ = 1/freq 
    delay_map = delays(length(cc), λ) .- λ*offset

    cc_filtered = smooth(cc)

    return ((τ) -> iswithin(τ, delay_map) ? cc_filtered[searchsortedfirst(delay_map, τ)] : 0) 
end 

function windowed_maximum_delay(f, g, freq, offset, freq_window, freq_start, freq_end, θ=constant(1)) 
    @assert (length(f) == length(g)) "Attempting to convolute signals with differing length"
    signal_length = length(f)
    pad_length = signal_length*2

    f_fft = pad_zero(f, pad_length) |> fft
    g_fft = pad_zero(g, pad_length) |> fft

    crosspow = g_fft .* conj(f_fft)

    λ = 1/freq
    delay = delays(length(crosspow), λ)

    max_delays = []
    weights = []

    for window_start ∈ range(freq_start, freq_end, freq_window)
        band = bandpass(window_start, window_start + freq_window, freq)
        conv = fftshift(abs.(ifft(crosspow .* band(crosspow) .* θ(crosspow))))

        max_index = argmax(conv)
        push!(max_delays, delay[max_index] - λ*offset)
        push!(weights, conv[max_index])
    end 
    return max_delays, weights
end 





function goertzel_hann(samples::Vector{Float64}, target_freq, sample_rate, window_length, start_at=1)
    p = 2π*round(Int, (target_freq*window_length)/sample_rate) / window_length
    a = 2/window_length
    s, c = sincos(p)
    d = 2*c

    hann_window = DSP.hann(window_length)

    z1, z2 = 0.0, 0.0

    r = range(start=start_at, length=window_length)
    window = samples[r] .* hann_window

    for sample ∈ window
        z0 = muladd(d, z1, sample-z2)
        z2 = z1
        z1 = z0
    end
    return muladd(z1, c, -z2)*a + *(z1, s, a)im
end

function goertzel_toa_estimate(samples::Vector{Float64}, freq, target_freq, offset; window_length=2^10, stride=1)
    r = 1:stride:(length(samples) - window_length)
    powers = map(r) do start_at
        abs(goertzel_hann(samples, target_freq, freq, window_length, start_at))
    end 
    return argmax(powers) + offset
end 





function maximum_delays(coords, speed_of_sound; inflate=1.0) 
    distance.(coords, permutedims(coords)) ./ speed_of_sound .* inflate
end 





function flatten(layers) 
    if isempty(layers)
        return [Int[]]
    end 

    tails = flatten(layers[2:end]) 
    heads = layers[1]

    if isempty(heads)
        return vcat.(missing, tails)
    else
        return [vcat(head, tail) for head ∈ heads for tail ∈ tails]
    end
end

function possible_distances(V, M)
    # delays = abs.(V .- V')
    # any(delays .> M)
    N = length(V)
    for i in 1:N, j in i+1:N
        if abs(V[i] - V[j]) > M[i,j]
            return false
        end
    end
    return true
end

function ambiguous_paths(candidate_paths)
    counts = Dict{Tuple, Int}()
    for cand in candidate_paths
        for (s, idx) in enumerate(cand)
            key = (s, idx)
            counts[key] = get(counts, key, 0) + 1
        end
    end

    ambiguous = Vector{Bool}(undef, length(candidate_paths))
    for (i, cand) in enumerate(candidate_paths)
        ambiguous[i] = any(counts[(s, cand[s])] > 1 for s in 1:length(cand))
    end

    return ambiguous
end


function candidate_event_paths(events, max_delays)
    
    next_event(events, ptrs) = findmin(getindex.(events, ptrs))
    stop(events, ptrs) = any(length.(events) .< ptrs) 

    N = length(events)
    ptrs = fill(1, N)
    candidate_paths = []

    

    while !stop(events, ptrs) 
        i_event, i_index = next_event(events, ptrs)
        ptrs[i_index] += 1

        candidates = Vector{Vector{eltype(i_event)}}(undef, N)
        # fill(eltype(i_event)[], N)
        candidates[i_index] = [i_event]

        for j_index ∈ 1:N 
            (j_index == i_index) && continue ## Skip i=j
            j_events = events[j_index][i_event .<= events[j_index] .<= i_event + max_delays[i_index, j_index]]
            candidates[j_index] = j_events 
        end


        for candidate ∈ flatten(candidates)
            if any(ismissing, candidate) 
                continue 
            end 
            if !possible_distances(candidate, max_delays)
                continue 
            end 
            push!(candidate_paths, candidate)
        end 
        
    end 

    ambiguous = ambiguous_paths(candidate_paths)
    return ambiguous, candidate_paths
end 


# function maximum_delays_samples(coords, speed_of_sound, freq) 
#     euclidean.(coords, permutedims(coords)) ./ speed_of_sound .* freq
# end 

# ## This is very Haskell-esque code, but all the copying and allocation 
# ## is insignificant at N=4
# function paths(layers) 
#     if isempty(layers)
#         return [Int[]]
#     end 

#     tails = paths(layers[2:end]) 
#     heads = layers[1]

#     if isempty(heads)
#         return vcat.(missing, tails)
#     else
#         return [vcat(head, tail) for head ∈ heads for tail ∈ tails]
#     end
# end


# function candidate_event_paths(events, min_support, max_delays)
    
#     next_event(events, ptrs) = findmin(getindex.(events, ptrs))
#     stop(events, ptrs) = any(length.(events) .== ptrs) 

#     N = length(events)
#     ptrs = fill(1, N)
#     candidate_paths = []

#     while !stop(events, ptrs) 
#         i_event, i_index = next_event(events, ptrs)
#         ptrs[i_index] += 1

#         candidates = fill(Int[], N)
#         candidates[i_index] = [i_event]
 
#         for j_index ∈ 1:N 
#             (j_index == i_index) && continue ## Skip i=j
#             j_events = events[j_index][i_event .<= events[j_index] .<= i_event + max_delays[i_index, j_index]]
#             candidates[j_index] = j_events 
#         end 


#         if count(c -> !isempty(c), candidates) >= min_support
#             push!(candidate_paths, paths(candidates)...)
#         end 
#     end 

#     return candidate_paths
# end 





# function goertzel_hann(samples::Vector{Float64}, target_freq, sample_rate, window_length, start_at=1)
#     p = 2π*round(Int, (target_freq*window_length)/sample_rate) / window_length
#     a = 2/window_length
#     s, c = sincos(p)
#     d = 2*c

#     hann_window = DSP.hann(window_length)

#     z1, z2 = 0.0, 0.0

#     r = range(start=start_at, length=window_length)
#     samples = samples[r] .* hann_window

#     for sample ∈ samples
#         z0 = muladd(d, z1, sample-z2)
#         z2 = z1
#         z1 = z0
#     end
#     return muladd(z1, c, -z2)*a + *(z1, s, a)im
# end

# function goertzel_toa_estimate(samples::Vector{Float64}, freq, target_freq, offset; window_length=2^10, stride=1)
#     powers = map(1:stride:(length(samples) - window_length)) do start_at
#         abs(goertzel_hann(samples, target_freq, freq, window_length, start_at))
#     end 
#     return argmax(powers) .+ offset
# end 


# function goertzel_toa_set(obs, candidate, freq, target_freq)
#     windows = event_windows(candidate, 2^12)

#     offsets = candidate .- minimum(candidate)

#     data = getrange.(obs, windows)
    
#     map(eachindex(data)) do i
#         goertzel_toa_estimate(data[i], freq, target_freq, offsets[i]) / freq
#     end 

# end 
