function [x_t, fs, t] = import_audio(filepath)
%   function [x_t, fs, t] = import_audio(filepath)
%
%   Import an audio signal from a wave file.
%
%   Parameters
%   ----------
%   filepath : string
%       path to a .wav file
%
%   Returns
%   -------
%   x_t : 1 x T array
%       time domain signal
%
%   t : 1 x T array
%       time points in seconds
%
%   fs : int
%       sample rate (samples per second)

%read audio file
[x_t, fs] = audioread(filepath);

%get dimensions
sizex_t = size(x_t);

%throw away additional channels
if sizex_t(2) > 1
    
    x_t = x_t(:,1);
    
end

%invert
x_t = x_t';

%vector of times
t = ((1:length(x_t))'/fs)';

end