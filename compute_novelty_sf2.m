% Nathan Lehrer N17119112
function [n_t_sf , t_sf, fs_sf] = compute_novelty_sf2(x_t, t, fs, win_size, hop_size)
    % Compute spectral flux novelty function.
    %
    % Parameters
    % ----------
    % x_t : 1 x T array
    % time domain signal
    % t : 1 x T array
    % time points in seconds
    % fs : int
    % sample rate of x t (samples per second)
    % win size : int
    % window size (in samples)
    % hop size : int
    % hop size (in samples)
    %
    % Returns
    % -------
    % n_t_sf : 1 x L array
    % novelty function
    % t_sf : 1 x L array
    % time points in seconds
    % fs_sf : float
    % sample rate of novelty function
    
    noverlap = win_size - hop_size;
    x_buff = buffer(x_t,win_size,noverlap,'nodelay');
    x_buff = x_buff(:,1:end-1); %remove zeros at the end
    frames = size(x_buff,2); 
    window = repmat(hamming(win_size),1,frames);
    
    X_spec = fft(x_buff .* window);
    X_mag_spec = abs(X_spec(1:(ceil((win_size+1)/2)),:));
    % spec flux frames-1 long since no useful info about deriv of frame 1
    X_mag_spec_flux = X_mag_spec(:,2:end) - X_mag_spec(:,1:end-1);
    n_t_sf = (2/win_size) * sum(((X_mag_spec_flux + abs(X_mag_spec_flux))/2).^2); 
    
    % first time pt is center of second window (after 1 hop)
    % last time is center of last window (after frames-1 hops)
    t_sf = linspace(((win_size-1)/2 + hop_size)/fs, ((win_size-1)/2 + ... 
        hop_size*(frames-1))/fs, frames-1);
    
    fs_sf = fs / hop_size; 
end