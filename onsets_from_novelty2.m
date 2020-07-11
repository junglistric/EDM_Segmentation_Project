% Nathan Lehrer N17119112
function [onset_a, onset_t, n_t_smoothed, thresh] = ...
    onsets_from_novelty2(n_t, t, fs, w_c, medfilt_len, offset)
    % Peak pick a novelty function.
    %
    % Parameters
    % ----------
    % n_t : 1 x L array
    % novelty function
    % t : 1 x L array
    % time points of n_t in seconds
    % fs : float
    % sample rate of n_t (samples per second)
    % w_c : float
    % cutoff frequency for Butterworth filter (Hz)
    % medfilt_len : int
    % Length of the median filter used in adaptive threshold. (samples)
    % offset : float
    % Offset in adaptive threshold.
    %
    % Returns
    % -------
    % onset_a : 1 x P array
    % onset amplitudes
    % onset_t : 1 x P array
    % time values of detected onsets (seconds)
    % n_t_smoothed : 1 x L array
    % novelty function after smoothing.
    % thresh : 1 x L array
    % adaptive threshold.
    
    L = length(n_t);
    
    [b,a] = butter(1,(2*w_c/fs)); %i.e. w_c/(fs/2)
    n_t_filt = filtfilt(b,a,n_t); 
    n_t_smoothed = n_t_filt/max(n_t_filt); %normalize
    
    thresh = zeros(1,L);
    %if medfilt_len odd, take (len-1)/2 samples before and after
    %if medfilt_len even, take (len)/2 samples before, (len/2)-1 after
    before = ceil((medfilt_len-1)/2); 
    after = floor((medfilt_len-1)/2);
    
    %for the first few samples, have to use slightly smaller median window
    for i = 1:before
        thresh(i) = median(n_t_smoothed(1:i+after));
    end
    for i = before+1:L-after
        thresh(i) = median(n_t_smoothed(i-before:i+after));
    end
    for i = L-after+1:L
        thresh(i) = median(n_t_smoothed(i-before:L));
    end
    thresh = thresh + offset;
    n_t_smoothed = n_t_smoothed-thresh;
    
    %Peak Picking
    n_deriv_neg = n_t_smoothed(3:end) - n_t_smoothed(2:end-1) < 0; 
    n_prev_deriv_pos = n_t_smoothed(2:end-1) - n_t_smoothed(1:end-2) > 0;
    onset_a = [];
    onset_t = [];
    %take value of after the pos deriv, not after the neg, 
    %i.e. first possible onset is at n_thresh(2)
    for i = 1:L-2
        if (n_deriv_neg(i) && n_prev_deriv_pos(i) && n_t_smoothed(i+1) > 0)
            onset_a = [onset_a n_t_smoothed(i+1)];
            onset_t = [onset_t t(i+1)];
        end
    end
end