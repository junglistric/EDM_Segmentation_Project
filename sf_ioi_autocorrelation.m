function [ioi,n_t_smoothed] = sf_ioi_autocorrelation(filename,sf_win_size,sf_hop_size, yin_win, yin_hop, min_lag,max_lag)
% Detect tempo using spectral flux and yin.
% 
% Zhiguang Eric Zhang N19320877
%
% Parameters
% ----------
% x_t : 1 x T array
%   time domain signal
% t : 1 x T array
%   time points in seconds
% fs : int
%   sample rate (samples per second)
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
%
% t_pitch : 1 x P array
%   time points in seconds

%import audio
[x_t,fs,t]=import_audio(filename);

%spectral flux to get onsets and onset sample times
[n_t_sf, t_sf, fs_sf] = compute_novelty_sf(x_t, t, fs, sf_win_size, sf_hop_size);
[onset_a, onset_t, n_t_smoothed, thresh] = ...
onsets_from_novelty(n_t_sf, t_sf, fs_sf, 11, 13, 0.02);

%onset_a_thresh = find(onset_a > .8);
%onset_t_thresh = onset_t(onset_a_thresh);

%detect syncopation
%onset_t_after = onset_t(onset_t > first_downbeat);
ioi = diff(onset_t);

%check max lag
if max_lag >= yin_win
    
    error('Maximum lag must be less than the window size.');

end

%lag vector
%lag = (0:max_lag)/fs_sf;

%segment the signal
y = buffer(ioi, yin_win, yin_win - yin_hop, 'nodelay');
onset_y = buffer(onset_t(2:end),yin_win,yin_win-yin_hop,'nodelay');

%output time vector
t_pitch = t_sf(1:yin_hop:end);

%pitch output vector
pitch = zeros(length(t_pitch),1);

%preallocate vector of YIN function and d_hat output
r_l = zeros(max_lag + 1, size(y,2));
d_hat = zeros(max_lag + 1, size(y,2));

%calculate RMS for suppression of silence
rms = sqrt(mean(y.^2));

%YIN
for j = 1:size(y,2)
    
        for l = min_lag:max_lag
    
            r_l(l+1,j) = sum((y(1:yin_win-l,j) - y(1+l:yin_win,j)).^2);
            
            %normalization
            if l == 0;
                
            d_hat(l+1,j) = 1;
            
            else
            
            d_hat(l+1,j) = (r_l(l+1,j)) / ((1/l)*(sum(r_l(1:l+1,j))));
            
            end
    
        end
        
end

%set everything before min_lag to 1
d_hat(1:min_lag,:) = 1;

%peak pick
for m = 1:size(d_hat,2);
    
    %invert
    d_hat_inv = 1.01*max(d_hat(:,m)) - d_hat(:,m);
    
    %normalize
    Y_min = d_hat_inv - min(d_hat_inv);
    YIN_norm = Y_min / max(Y_min);
    YIN_norm(1:min_lag) = 0;

    [~,b]=findpeaks(YIN_norm,'MINPEAKHEIGHT',.5*max(YIN_norm));
    
    lag = onset_y(:,m);
    
    if length(b) >= 3 && rms(m) > 0.02
        
        pitch(m) = 240/(lag(b(3)) - lag(b(2)));
        
    elseif length(b) == 2 && rms(m) > 0.02
        
        pitch(m) = 240/(lag(b(2)) - lag(b(1)));
        
    elseif length(b) == 1 && rms(m) > 0.02
        
        pitch(m) = 240/(lag(b(1)));   
        
    elseif rms(m) <= 0.02
        
        pitch(m) = 0;

    end

end

plot(t_pitch,pitch);
disp(mode(pitch));

end