% Nathan Lehrer N17119112
function [tempogram, bpm, t_tempogram, t_nov] = compute_tempogram2(x_t, t, fs, ...
    win_size_nov, win_size_tpo, min_bpm, max_bpm)
    % Compute a tempogram
    % 
    % Parameters
    % ----------
    % x_t : 1 x T array
    %     time domain signal
    % t : 1 x T array
    %     time points in seconds
    % fs : int
    %     sample rate of x_t (samples per second)
    % win_size_nov : int
    %     window size for the novelty function (in samples)
    % win_size_tpo : int
    %     window size for the tempogram (in samples)
    % min_bpm : int
    %     minimum tempo value (beats per minute)
    % max_bpm : int
    %     maximum tempo value (beats per minute)
    % Returns
    % -------
    % tempogram : NT x NF array
    %     complex valued tempogram
    % bpm : 1 x NF array
    %     frequency points in beats per minute
    % t_tempogram : 1 x NT array
    %     time values of the tempogram in seconds
    % t_nov : 1 x NNT array
    %     time values of the novelty function in seconds
    
    % Compute novelty function
    hop_size_nov = win_size_nov / 2;
    [n_t,t_nov,fs_nov] = compute_novelty_sf2(x_t, t, fs, win_size_nov, hop_size_nov);
    
    % Post process nov func to smooth and normalize it
    w_c = 1;
    medfilt_len = 12;
    offset = 0;
    [~, ~, n_t_smoothed, ~] = ...
        onsets_from_novelty2(n_t, t_nov, fs_nov, w_c, medfilt_len, offset);
    
    % Compute tempogram
    hop_size_tpo = win_size_tpo / 2;
    noverlap = win_size_tpo - hop_size_tpo;
    n_buff = buffer(n_t_smoothed,win_size_tpo,noverlap,'nodelay');
    n_buff = n_buff(:,1:end-1); %remove zeros at the end
    frames = size(n_buff,2);
    
    window = hamming(win_size_tpo);
    n_final = n_buff .* repmat(window,[1 size(n_buff,2)]); 
    
    min_bpm_hz = min_bpm/60;
    max_bpm_hz = max_bpm/60;
    min_bin = floor(min_bpm_hz/fs_nov * win_size_tpo); 
    max_bin = ceil(max_bpm_hz/fs_nov * win_size_tpo); 
        %each entry is row num * column num
    NF = 100;
    bins = linspace(min_bin,max_bin,NF);
    bpm = bins * (fs_nov / win_size_tpo) * 60;
    nk_array = bins' * (0:win_size_tpo-1);
        %kth row of s_star = s*_k basis vec 
    s_star = exp(-1j*2*pi*nk_array/win_size_tpo);
    tempogram = (s_star * n_final)'; 
    
    t_tempogram = linspace(((win_size_tpo-1)/2)/fs_nov, ((win_size_tpo-1)/2 +  ...
        hop_size_tpo*(frames-1))/fs_nov, frames);
end