function [output,ph,t_tempo,tempo] = sf_autocorrelation_phase(filename,sf_win_size,sf_hop_size, yin_win, yin_hop, min_lag,max_lag)
% Detect tempo using spectral flux and yin.
% 
% Zhiguang Eric Zhang N19320877
%
% Parameters
% ----------
% filename : audio file path
%   input file name
% sf_win_size : int
%   novelty window size (in samples)
% sf_hop_size : int
%   novelty hop size (in samples)
% yin_win : int
%   YIN window size
% yin_hop : int
%   YIN hop size
% min_lag : int
%   minimum possible lag value (in samples)
% max_lag : int
%   maximum possible lag value (in samples)
%
% Returns
% -------
% tempo : 1 x P array
%   detected tempo values (bpm)
% t_tempo : 1 x P array
%   time points in seconds
% ph : 1 x P array
%   detected phase
% output : 1 x 1 array
%   mode of tempo (bpm)

%import audio
[x_t,fs,t]=import_audio(filename);

%spectral flux to get onsets and onset sample times
[n_t_sf, t_sf, fs_sf] = compute_novelty_sf(x_t, t, fs, sf_win_size, sf_hop_size);
[onset_a, onset_t, n_t_smoothed, testthresh] = ...
onsets_from_novelty(n_t_sf, t_sf, fs_sf, 11, 13, 0.02);

%kick drum bandpass filter from 50Hz to 150Hz
%[b,a] = butter(1,[50 150]/(fs/2),'bandpass');
%x_t_filt = filtfilt(b,a,x_t);
%spectral flux to get onsets and onset sample times
%[n_t_sf2, t_sf2, fs_sf2] = compute_novelty_sf(x_t_filt, t, fs, sf_win_size, sf_hop_size);
%[onset_a2, onset_t2, n_t_smoothed2, abovethresh] = ...
%onsets_from_novelty(n_t_sf2, t_sf2, fs_sf2, 11, 13, 0.02);
%detect syncopation
%onset_t_after = onset_t2(onset_t2 > first_downbeat);

%ioi = diff(onset_t_after);

%check max lag
if max_lag >= yin_win
    
    error('Maximum lag must be less than the window size.');

end

%lag vector
lag = (0:max_lag)/fs_sf;

%segment the signal
y = buffer(n_t_smoothed, yin_win, yin_win - yin_hop, 'nodelay');

%output time vector
t_tempo = t_sf(1:yin_hop:end);

%pitch and phase output vector
tempo = zeros(length(t_tempo),1);
ph = zeros(length(t_tempo),1);

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
    
    [~,b]=findpeaks(YIN_norm,'MINPEAKHEIGHT',.9*max(YIN_norm));
    
    if length(b) >= 3 && rms(m) > 0.02
        
        tempo(m) = 240/(lag(b(3)) - lag(b(2)));
        
    elseif length(b) == 2 && rms(m) > 0.02
        
        tempo(m) = 240/(lag(b(2)) - lag(b(1)));
        
    elseif length(b) == 1 && rms(m) > 0.02
        
        tempo(m) = 240/lag(b(1));
        
    elseif rms(m) <= 0.02
        
       	tempo(m) = 0;

    end

    %decimation of octave errors
    if tempo(m) >= 500
        
        tempo(m) = tempo(m) / 4;
    
    elseif tempo(m) > 200 && tempo(m) < 500
        
        tempo(m) = tempo(m) / 2;
       
    end

end

%cross correlation phase initialization
r_ph = zeros(yin_win,size(y,2));
impulse_train = zeros(yin_win,1);

%cross correlation
for j = 1:size(y,2)

    if tempo(j) == 0
        continue;
    end
    beat_period = 1/(tempo(j)/60);
    impulse_train(1:round(beat_period*fs_sf):end) = 1;

    for l = 1:size(impulse_train,1)

        r_ph(l,j) = (1/(yin_win-l)) * sum(y(1:yin_win-l,j) .* impulse_train(l+1:yin_win));
    
    end

    %get the phase
    ph(j) = max(r_ph(:,j));

end

output = mode(tempo);
%plot(t_tempo,ph);
disp(output);

end