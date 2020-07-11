% Nathan Lehrer N17119112
function [pitch, t_pitch] = detect_pitch_acf(x_t, t, fs, win_size, ...
    hop_size, min_lag, max_lag)
    % Compute pitches and pitch locations using Spectral ACF
    %
    % Parameters
    % ----------
    % x_t : 1 x T array
    %   time domain signal
    % t : 1 x T array
    %   time points in seconds
    % fs : int
    %   sample rate of x_t (samples per second)
    % win_size : int
    %   window size (in samples)
    % hop_size : int
    %   hop size (in samples)
    % min_lag : int
    %   minimum possible lag value (in samples)
    % max_lag : int
    %   maximum possible lag value (in samples)
    %
    % Returns
    % -------
    % pitch : 1 x P array
    %   detected pitch values (Hz)
    % t_pitch : 1 x P array
    %   time points in seconds
    
    % Compute spectrogram
    noverlap = win_size - hop_size;
    x_buff = buffer(x_t,win_size,noverlap,'nodelay');
    x_buff = x_buff(:,1:end-1); %remove zeros at the end
    frames = size(x_buff,2); 
    window = repmat(hamming(win_size),1,frames);
    X_spec = fft(x_buff .* window);
    X_mag_spec = abs(X_spec(1:(ceil((win_size+1)/2)),:));
    
    % Compute spectral acf function
    lags = min_lag:max_lag;
    L = length(lags);
    rx = zeros(L,frames);
    N = size(X_mag_spec,1);    
    for i = 1:L
        lf = lags(i);
        % rx holds spectral acf of ith frame in ith column
        rx(i,:) = (1/(N-lf)) * sum(X_mag_spec(1:N-lf,:).*X_mag_spec(lf+1:N,:));
    end
        
    % Butterworth filter
    w_c = 0.35; %remove fast fluctuations in acf function
    [b,a] = butter(1,w_c); 
    rx_filt = filtfilt(b,a,rx);
 
    % Median filter
    medfilt_len = 12;
    offset = 0.14;
    before = ceil((medfilt_len-1)/2); 
    after = floor((medfilt_len-1)/2);
    thresh = zeros(L,frames);
        %thresh behaves differently at beginning and end
    for i = 1:before
         thresh(i,:) = median(rx_filt(1:i+after,:));
    end
    for i = before+1:L-after
        thresh(i,:) = median(rx_filt(i-before:i+after,:));
    end
    for i = L-after+1:L
        thresh(i,:) = median(rx_filt(i-before:L,:));
    end
    thresh = thresh + offset;
    rx_filt = rx_filt - thresh;
      
    % Peak picking 
    rx_deriv_neg = rx_filt(3:end,:) - rx_filt(2:end-1,:) < 0; 
    rx_prev_deriv_pos = rx_filt(2:end-1,:) - rx_filt(1:end-2,:) > 0;
    rx_pos = rx_filt(3:end,:) > 0;
    rx_peaks = rx_deriv_neg & rx_prev_deriv_pos & rx_pos;
        % Add a row of zeros to make the row indices line up with rx_filt
    rx_peaks = [zeros(1,frames) ; rx_peaks];
    peak_idx = find(rx_peaks);
    [subrows,subcols] = ind2sub([L-1 frames],peak_idx);
    
    % Calculate time and pitch vectors
    t_pitch = linspace(((win_size-1)/2)/fs, ((win_size-1)/2 + ... 
        hop_size*(frames-1))/fs, frames);   
    pitch = zeros(1,frames);
    for j = 1:frames
        subrows_frame = subrows(subcols == j);
        subcols_frame = subcols(subcols == j);
        if numel(subcols_frame) == 0
          pitch(j) = 0; %by convention, 0 means no pitch present
        else
          pitch(j) = lags(subrows_frame(1))*(fs/win_size); %pick first peak
        end
    end
end
    