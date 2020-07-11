function [beat_times, beat_idx] = beats_from_plp(plp, t_nov)
% Select beats from PLP curve.
%
% Zhiguang Eric Zhang N19320877
%
% Parameters
% ----------
% plp : 1 x NNT array
%   PLP curve
% t_nov : 1 x NNT array
%   time values of the novelty function (seconds)
%
% Returns
% -------
% beat_times : 1 x B array
%   time values of detected beats (seconds)
% beat_idx : 1 x B array
%   indices of detected beats (samples)

%find beat indices
[~,beat_idx]=findpeaks(plp);

%find beat times
beat_times = t_nov(beat_idx);

end