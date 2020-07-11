function [mfccs, fs_mfcc] = compute_mfccs(filepath, win_size, hop_size, ...
min_freq, max_freq, num_mel_filts, n_dct)
% Compute MFCCs from audio file.
%
% Parameters
% ----------
% filepath : string
%   path to .wav file
% win_size : int
%   spectrogram window size (samples)
% hop_size : int
%   spectrogram hop size (samples)
% min_freq : float
%   minimum frequency in Mel filterbank (Hz)
% max_freq : float
%   maximum frequency in Mel filterbank (Hz)
% num_mel_filts: int
%   number of Mel filters
% n_dct: int
%   number of DCT coefficients
%
% Returns
% -------
% mfccs : n_dct x NT array
%   MFCC matrix (NT is number spectrogram frames)
% fs_mfcc : int
%   sample rate of MFCC matrix (samples/sec)

%import audio
[wave,fs]=audioread(filepath);

%throw away one channel
wave = wave(:,1);

%pad the signal
wave=[zeros(hop_size,1);wave;zeros(hop_size,1)];

%fs_mfcc
fs_mfcc = fs/hop_size;

%hamming window
win = hamming(win_size);

%spectrogram and power spectrum
S = spectrogram(wave,win,win_size-hop_size,win_size,fs);
S_pow = 20*log10(abs(S));

%mel filterbank init
min_mel = hz2mel(min_freq);
max_mel = hz2mel(max_freq);
mels = linspace(min_mel,max_mel,num_mel_filts);
delta_mel = mels(2)-mels(1);
mels = [mels(1)-delta_mel,mels,mels(num_mel_filts)+delta_mel];
center_freqs = mel2hz(mels);

%Hz indices of each bin
res = fs/win_size;
bins = (0:round(win_size/2));
hz_indices = bins*res;

%indices of center freqs
center_bins = find_nearest(hz_indices',center_freqs);

%allocate Mel filterbank
mel_filt_bank = zeros(round(win_size/2)+1,num_mel_filts);

%triangular windows filterbank
for t = 1:num_mel_filts    
    
    win_left = triang(2*(length(center_bins(t):center_bins(t+1)))-2);
    win_right = triang(2*(length(center_bins(t+1)+1:center_bins(t+2)))-2);
    
    mel_filt_bank(center_bins(t)+1:center_bins(t+1),t) = win_left(1:round(length(win_left)/2));
    mel_filt_bank(center_bins(t+1)+1:center_bins(t+2)-1,t) = win_right(round(length(win_right)/2)+1:end);
    
    %normalize to unity sum
    mel_filt_bank(:,t) = mel_filt_bank(:,t)/sum(mel_filt_bank(:,t));
    
end

%plot(mel_filt_bank);

%Mel spectrum
S_Mel = S_pow' * mel_filt_bank;

%DCT
S_dct = dct(S_Mel');
mfccs = S_dct(1:n_dct,:);
mfccs = flipud(mfccs);

end