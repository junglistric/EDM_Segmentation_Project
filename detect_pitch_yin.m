% Nathan Lehrer N17119112
function [pitch, t_pitch] = detect_pitch_yin(x_t, t, fs, win_size, ...
    hop_size, min_lag, max_lag)
    % Detect pitches using yin
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
    
    % Split x_t into frames
    noverlap = win_size - hop_size;
    x_buff = buffer(x_t,win_size,noverlap,'nodelay');
    x_buff = x_buff(:,1:end-1); %remove zeros at the end
    frames = size(x_buff,2); 
    
    % Compute yin squared error function
    lags = min_lag:max_lag;
    L = length(lags);
    N = win_size;
    d = zeros(L,frames);
    dhat = zeros(L,frames);
    for i = 1:L
        lf = lags(i);
        % dhat holds yin squared error of ith frame in ith column
        d(i,:) = sum((x_buff(1:N-lf,:)-x_buff(lf+1:N,:)).^2);
        if i == 1
            dhat(i,:) = 1;
        else
            dhat(i,:) = d(i,:) ./ mean(d(1:i,:));
        end
    end
    
    % Peak pick
    dhat_deriv_pos = dhat(3:end,:) - dhat(2:end-1,:) > 0; 
    dhat_prev_deriv_neg = dhat(2:end-1,:) - dhat(1:end-2,:) < 0;
    dhat_under = dhat(3:end,:) < 0.45;
    dhat_troughs = dhat_deriv_pos & dhat_prev_deriv_neg & dhat_under;
        % Add a row of zeros to make the row indices line up with dhat
    dhat_troughs = [zeros(1,frames) ; dhat_troughs];
    trough_idx = find(dhat_troughs);
    [subrows,subcols] = ind2sub([L-1 frames],trough_idx);

    % Calculate time and pitch vectors
    t_pitch = linspace(((win_size-1)/2)/fs, ((win_size-1)/2 + ... 
        hop_size*(frames-1))/fs, frames);
    pitch = zeros(1,frames);
    for j = 1:frames
        subrows_frame = subrows(subcols == j);
        subcols_frame = subcols(subcols == j);
        
        % Remove spurious peaks in quick succession
        mark_for_del = [];
        for i = 1:length(subrows_frame)-1
            if lags(subrows_frame(i+1)) - lags(subrows_frame(i)) < 18
                [~,index] = max(subrows_frame(i:i+1));
                mark_for_del = [mark_for_del i-1+index];
            end
        end
        subrows_frame(mark_for_del) = [];
        subcols_frame(mark_for_del) = [];
        
        if numel(subcols_frame) == 0
            pitch(j) = 0; %by convention, 0 means no pitch present
        else
            if numel(subcols_frame) == 1
                optim_lag = lags(subrows_frame(1));
            else
                candidate_lags = lags(subrows_frame(1:2));
                candidate_vals = dhat(subrows_frame(1:2),j);
                % if they are close together in value, pick first one
                if abs(candidate_vals(1) - candidate_vals(2)) < 0.025  
                    optim_lag = candidate_lags(1);
                % otherwise, pick the minimum of the first two
                else
                    [~,index] = min(candidate_vals(1:2));
                    optim_lag = candidate_lags(index);
                end
            end
            pitch(j) = fs/optim_lag; 
        end
    end  
end