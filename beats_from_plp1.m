% Nathan Lehrer N17119112
function [beat_times, beat_idx] = beats_from_plp(plp, t_nov)
    % Select beats from PLP curve
    % 
    % Parameters
    % ----------
    % plp : 1 x NNT array
    %     PLP curve
    % t_nov : 1 x NNT array
    %     time values of the novelty function (seconds)
    % 
    % Returns
    % -------
    % beat_times : 1 x B array
    %     time values of detected beats (seconds)
    % beat_idx : 1 x B
    %     indices of detected beats (samples)
   
    p_deriv_neg = (plp(3:end) - plp(2:end-1)) < 0; 
    p_prev_deriv_pos = (plp(2:end-1) - plp(1:end-2)) > 0;
    
    beat_idx = find(p_deriv_neg & p_prev_deriv_pos) + 1;
    beat_times = t_nov(beat_idx);
end
    