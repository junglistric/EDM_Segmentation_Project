function [n_t_sf, t_sf, fs_sf] = compute_novelty_sf(x_t, t, fs, win_size, hop_size)
%   function [n_t_sf, t_sf, fs_sf] = compute_novelty_sf(x_t, t, fs, win_size, hop_size)
%
%   Compute spectral flux novelty function.
%
%   Parameters
%   ----------
%   x_t : 1 x T array
%       time domain signal
% 
%   t : 1 x T array
%       time points in seconds
%
%   fs : int
%   sample rate of x_t (samples per second)
%
%   win_size : int
%       window size (in samples)
%
%   hop_size : int
%       hop size (in samples)
%
%   Returns
%   -------
%   n_t_sf : 1 x L array
%       novelty function
%
%   t_sf : 1 x L array
%       time points in seconds
%
%   fs_sf : float
%       sample rate of novelty function

%zero pad a window at the beginning
x_pad = [zeros(win_size,1);x_t'];

%segment signal with padding
Y = buffer(x_pad,win_size,win_size-hop_size,'nodelay');

%sample rate of the function
fs_sf = 1/(hop_size/fs);

%time vector of novelty function making sure the first output novelty value
%starts at ~time zero
t_neg = win_size / fs;
t_neg_vec = (-t_neg:1/fs:0);
t = [t_neg_vec,t];
t_sf = t(win_size+1:hop_size:end);

%window and frequency domain magnitudes and take DC to Nyquist
window = hann(win_size);
Y = bsxfun(@times,window,Y);
Y_f = abs(fft(Y));
Y_f = Y_f(1:round(size(Y_f,1)/2)+1,:);

%spectral flux
SF = Y_f(:,2:end) - Y_f(:,1:end-1);

%half wave rectify
SF = (SF+abs(SF))/2;

%sum and normalize
n_t_sf = sum(SF) / length(SF);

end