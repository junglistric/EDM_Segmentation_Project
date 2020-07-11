% Nathan Lehrer N17119112
function plp = compute_plp(phi, omega, win_size_tpo, fs_nov, t_nov)
    % Compute PLP curve.
    %
    % Parameters
    % ----------
    % phi : 1 x NT array
    %     phases of maximum tempogram values over time
    % omega : 1 x NT array
    %     frequencies (beats per minute) of maximum tempogram values over time
    % win_size_tpo : int
    %     window size for the tempogram (in samples)
    % fs_tnov : float
    %     sample rate of the novelty function (samples/second)
    % t_nov : 1 x NNT array
    %     Time values of the novelty function (seconds)
    %     
    % Returns
    % -------
    % plp : 1 x NT array
    %     PLP curve
    
    NT = length(omega);
    NNT = length(t_nov);
    hop_size_tpo = win_size_tpo / 2;
    win = hamming(win_size_tpo);
    windows = zeros(NT,NNT);
    cosines = zeros(NT,NNT);
    omega_hz = omega/60;
    local_time = (0:win_size_tpo-1)/fs_nov;
    for i = 1:NT
        windows(i, (i-1)*hop_size_tpo + 1 : (i-1)*hop_size_tpo + win_size_tpo) = win;
        cosines(i, (i-1)*hop_size_tpo + 1 : (i-1)*hop_size_tpo + win_size_tpo) ...
            = cos(2*pi*(omega_hz(i)*local_time - phi(i))); 
    end
    kernels = windows .* cosines; 
    % Each row contains a cosine kernel of a different freq/phase
    %kernels = windows .* cos(2*pi*(omega_hz'*t_nov - repmat(phi',[1 NNT])));
    plp = sum(kernels);
    plp(plp<0)=0; % half wave rectify
end