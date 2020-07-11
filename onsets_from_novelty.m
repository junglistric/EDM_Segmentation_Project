function [onset_a, onset_t, n_t_smoothed, thresh] = ...
onsets_from_novelty(n_t, t, fs, w_c, medfilt_len, offset)
%   function [onset_a, onset_t, n_t_smoothed, thresh] = ...
%   onsets_from_novelty(n_t, t, fs, w_c, medfilt_len, offset)
%
%   Peak pick a novelty function.
%
%   Parameters
%   ----------
%   n_t : 1 x L array
%       novelty function
%   t : 1 x L array
%       time points of n_t in seconds
%   fs : float
%       sample rate of n_t (samples per second)
%   w_c : float
%       cutoff frequency for Butterworth filter (Hz)
%   medfilt_len : int
%       Length of the median filter used in adaptive threshold. (samples)
%   offset : float
%       Offset in adaptive threshold.
%
%   Returns
%
%   -------
%   onset_a : 1 x P array
%       onset amplitudes
%   onset_t : 1 x P array
%       time values of detected onsets (seconds)
%   n_t_smoothed : 1 x L array
%       novelty function after smoothing.
%   thresh : 1 x L array
%       adaptive threshold.

%normalized cutoff freq
Wn = w_c/(fs/2);

%butterworth coefficients
[B,A] = butter(1,Wn,'low');

%zero phase filter novelty function
Y = filtfilt(B,A,n_t);

%normalize filtered novelty function
Y_min = Y - min(Y);
n_t_smoothed = Y_min / max(Y_min);

%parameter L
if mod(medfilt_len,2) == 0
    
    L1 = (medfilt_len/2);
    L2 = (medfilt_len/2);
    
elseif mod(medfilt_len,2) == 1
   
    L1 = floor(medfilt_len/2);
    L2 = ceil(medfilt_len/2);
    
end

%parameter B
B = 1;

%pad novelty function
n_t_smoothed_pad = [zeros(B*L1,1);n_t_smoothed';zeros(L2,1)];

%preallocate adaptive threshold
thresh = zeros(length(n_t),1);

%adaptive filter with B, L
for k = B*L1+1:length(n_t_smoothed_pad)-L2
    
    thresh(k-(B*L1)) = median(n_t_smoothed_pad(k-(B*L1):k+L2)) + offset;
    
end
thresh = thresh';

%get derivative of smoothed novelty function above threshold
peaksdiff = diff(n_t_smoothed);

%find zero crossings of first order derivative
peaksdiff = [0;peaksdiff'];
zerocrossings = sign(peaksdiff(1:end-1)) - sign(peaksdiff(2:end));
zerocrossings = [zerocrossings;0];

%get positive peaks
A = find(zerocrossings >= 1);

%smoothed novelty function above threshold and half wave rectify to only
%get positive peaks
abovethresh = n_t_smoothed - thresh;
signthresh = sign(abovethresh);
newthresh = (signthresh + abs(signthresh))/2;

%get amplitude of onsets
onset_a = n_t_smoothed(A) .* newthresh(A);
onset_a(onset_a == 0) = [];

%get times of onsets and preserve onset at time zero if there is one
%detected
timezero = false;
onset_t = t(A) .* newthresh(A);
if newthresh(A(1)) == 1 && t(A(1)) == 0;
    timezero = true;
end
onset_t(onset_t == 0) = [];

%if there was an onset at time zero, replace the value
if timezero == true
    
    onset_t = [0,onset_t];
    
end

end