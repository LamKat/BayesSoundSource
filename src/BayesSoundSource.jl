module BayesSoundSource

const SPEED_OF_SOUND = 343.0

include("Util.jl")
export  distance, 
        pairwise_difference_matrix,
        num_pairwise_differences,
        synthetic_tdoa,
        synthetic_toa,
        stochastic_trajectory

include("Freq.jl")
# export

include("Models.jl")
export  FlatTrajPrior,  
        GPTrajPrior,
        toa_tdoa_model,
        tdoa_model,
        toa_model, 
        sample_traj, 
        traj_length_from_chain,
        map_estimate, 
        spread

include("TimeDelayEst.jl")
export  GCC_maximum, 
        GCC, 
        PHAT, 
        bandpass, 
        candidate_event_paths, 
        maximum_delays



include("GDOP.jl")
export  crlb_toa, 
        crlb_tdoa, 
        gdop_toa,
        gdop_tdoa

end # module BayesSoundSource
