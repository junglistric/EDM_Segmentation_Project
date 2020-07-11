function [novelty_t]=mfcc_sim_measure(filename,first_downbeat,tempo,tolerance,sf_win_size,sf_hop_size, min_freq, max_freq, num_mel_filts, n_dct)
%measure-synchronous MFCC self-similarity segmentation

%get measure boundaries
[output] = downbeat_track_input(filename,first_downbeat,tempo,tolerance,sf_win_size,sf_hop_size);

%get info
[wave,fs]=audioread(filename);
wave=wave(:,1);

%sample measure boundaries and pad
measures = round(output * fs);
measures = [1;measures;length(wave)];
t_measures = measures/fs;

%find longest 'measure'
measure_diff = diff(measures);
[a,b] = max(measure_diff);

%allocate output matrix
segments = zeros(a,length(measures)-1);

%segment the track
for m = 1:size(segments,2)
    
    %disp(size(segments(1:measure_diff(m)-1,m)));
    %disp(size(wave(measures(m):measures(m+1)-1)));
    
    segments(1:measure_diff(m),m) = segments(1:measure_diff(m),m) + wave(measures(m):measures(m+1)-1);
    
end

%compute spectrogram
win = hamming(a);
segments = bsxfun(@times,win,segments);

disp('here3');
disp(a);

%preallocate M
M = zeros(a,a);

%complex conjugate basis function-derived matrix
for k = 0:a-1
    
    for m = 0:a-1
        
        M(m+1,k+1) = conj(cos(2*pi*m*k/a)+1i*(sin(2*pi*m*k/a)));
        
    end

end

disp('here1');

%inner product matrix multiplication
S = M * segments;

%truncate S between DC and Nyquist
S = S(1:size(S,1)/2+1,1:end-1);
S_pow = 20*log10(abs(S));

%mel filterbank init
min_mel = hz2mel(min_freq);
max_mel = hz2mel(max_freq);
mels = linspace(min_mel,max_mel,num_mel_filts);
delta_mel = mels(2)-mels(1);
mels = [mels(1)-delta_mel,mels,mels(num_mel_filts)+delta_mel];
center_freqs = mel2hz(mels);

%Hz indices of each bin
res = fs/a;
bins = (0:round(a/2));
hz_indices = bins*res;

%indices of center freqs
center_bins = find_nearest(hz_indices',center_freqs);

%allocate Mel filterbank
mel_filt_bank = zeros(round(a/2)+1,num_mel_filts);

%triangular windows filterbank
for t = 1:num_mel_filts    
    
    win_left = triang(2*(length(center_bins(t):center_bins(t+1)))-2);
    win_right = triang(2*(length(center_bins(t+1)+1:center_bins(t+2)))-2);
    
    mel_filt_bank(center_bins(t)+1:center_bins(t+1),t) = win_left(1:round(length(win_left)/2));
    mel_filt_bank(center_bins(t+1)+1:center_bins(t+2)-1,t) = win_right(round(length(win_right)/2)+1:end);
    
    %normalize to unity sum
    mel_filt_bank(:,t) = mel_filt_bank(:,t)/sum(mel_filt_bank(:,t));
    
end

disp('here2');

%Mel spectrum
S_Mel = S_pow' * mel_filt_bank;

%DCT
S_dct = dct(S_Mel');
mfccs = S_dct(1:n_dct,:);
mfccs = flipud(mfccs);

%features
%[t_features,fs_features, features] = compute_features(t_mfcc,mfccs, fs_mfcc);

%allocate normalized matrix
mfccs_norm = zeros(size(mfccs));

%L2 norm
for l = 1:size(mfccs,2)
    
    mfccs_norm(:,l) = mfccs(:,l) / norm(mfccs(:,l));
    
end

%compute similarity using Euclidean distance
output = zeros(length(mfccs_norm),length(mfccs_norm));
for l = 1:length(mfccs_norm)
    
    for m = 1:length(mfccs_norm)
        
        output(m,l) = norm(mfccs_norm(:,m) - mfccs_norm(:,l));
        
    end
     
end

%100 x 100 checkerboard kernel
checker = zeros(100,100);
one = ones(50,50);
negone = one .* -1;
checker(1:50,1:50) = negone;
checker(51:end,1:50) = one;
checker(1:50,51:end) = one;
checker(51:end,51:end) = negone;

%correlate similarity matrix with kernel
for m = 1:length(output)-length(checker)+1
    
    nov(m) = sum(sum(output(m:m+length(checker)-1,m:m+length(checker)-1) .* checker));
    
end

%time vector
t_features = t_measures(50:end-50);

%peak pick
%[onset_a, onset_t, n_t_smoothed, thresh] = ...
%onsets_from_novelty(nov, t_features, fs_features, 20, 50, .02);

%normalized cutoff freq
%Wn = 10/(fs_features/2);

%butterworth coefficients
%[B,A] = butter(1,Wn,'low');

%zero phase filter novelty function
%Y = filtfilt(B,A,nov);

[c,d]=findpeaks(nov);
novelty_t=t_features(d);

plot(t_features,nov);

end