% Zhiguang Eric Zhang N19320877

function [onset_t]=mfcc_sim(filename,win_size,hop_size, min_freq, max_freq, num_mel_filts, n_dct)
%MFCC self-similarity segmentation

%compute mfccs
[t_mfcc,mfccs, fs_mfcc] = compute_mfccs(filename, win_size, hop_size, ...
min_freq, max_freq, num_mel_filts, n_dct);

%features
[t_features,fs_features, features] = compute_features(t_mfcc,mfccs, fs_mfcc);

%allocate normalized matrix
mfccs_norm = zeros(size(features));

%L2 norm
for l = 1:size(features,2)
    
    mfccs_norm(:,l) = features(:,l) / norm(features(:,l));
    
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
t_features = t_features(50:end-50);

%peak pick
%[onset_a, onset_t, n_t_smoothed, thresh] = ...
%onsets_from_novelty(nov, t_features, fs_features, 20, 50, .02);

[a, b]=findpeaks(nov);
onset_t=t_features(b);

plot(t_features,nov);

end

end
