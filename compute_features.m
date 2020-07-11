function features = compute_features(mfccs, fs_mfcc)
% Compute features from MFCCs.
%
% Parameters
% ----------
% mfccs : n_dct x NT matrix
%   MFCC matrix
% fs_mfcc : int
%   sample rate of MFCC matrix (samples/sec)
%
% Returns
% -------
% features: NF X NE
%   matrix of segmented and averaged MFCCs
%   (NF is number of features = n_dct-1 and
%   NE is number of examples)

%remove the first coefficient
mfccs = mfccs(end:-1:1,:);
mfccs = mfccs(2:end,:);

%number of windows in each 1 second sequence
wins_per_sec = round(fs_mfcc);

%number of examples
NE = ceil(size(mfccs,2)/wins_per_sec);

%allocate features vector
features = zeros(size(mfccs,1),NE);

%starting window
starting_win = 1;

%features
for m = 1:NE
    
    for l = 1:size(mfccs,1)
        
        if m == NE
            
            features(l,m) = mean(mfccs(l,starting_win:end));
        
        else
    
            features(l,m) = mean(mfccs(l,starting_win:starting_win+wins_per_sec));
        
        end
    
    end
    
    starting_win = starting_win + wins_per_sec;
    
end

end