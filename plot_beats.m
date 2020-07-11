% Nathan Lehrer N17119112
function beat_times = plot_beats(filepath, win_size_nov, win_size_tpo, min_bpm, max_bpm)
    % Compute and plot tempogram, novelty function, and detected beats.
    % 
    % Parameters
    % ----------
    % filepath : string
    %     path to .wav file
    % win_size_nov : int
    %     window size for the novelty function (in samples)
    % win_size_tpo : int
    %     window size for the tempogram (in samples)
    % min_bpm : int
    %     minimum tempo value (beats per minute)
    % max_bpm : int
    %     maximum tempo value (beats per minute)
    %     
    % Returns
    % -------
    % beat_times : 1 x B array
    %     time values of detected beats (seconds)

    [x_t, fs, t] = import_audio(filepath);
    
    hop_size_nov = win_size_nov / 2;
    [n_t,t_nov,fs_nov] = compute_novelty_sf(x_t, t, fs, win_size_nov, hop_size_nov);
    
    [tempogram, bpm, t_tempogram, t_nov] = compute_tempogram(x_t, t, fs, ...
        win_size_nov, win_size_tpo, min_bpm, max_bpm);

    [phi_hat, omega_hat, omega_idx] = max_tempogram(tempogram, bpm);

    phi = phi_hat;
    omega = omega_hat;
    plp = compute_plp(phi, omega, win_size_tpo, fs_nov, t_nov);
    
    [beat_times, beat_idx] = beats_from_plp(plp, t_nov);
    
    subplot(2,1,1)
    imagesc(t_tempogram,bpm,abs(tempogram'));
    set(gca,'YDir','normal'); % Y=0 at bottom
    xlabel('time, sec')
    ylabel('local tempo, bpm')
    
    subplot(2,1,2)
    plot(t_nov,n_t/max(n_t))
    xlabel('time, sec')
    ylabel('novelty function and beats')
    hold on;
    for i = 1:length(beat_times);
        line([beat_times(i) beat_times(i)],[0 1],'Color','gr');
    end
    hold off;
    
    %Sonify beats (optional)
     sonified_beats = zeros(length(x_t),1);
     beat_idx_fs = round(beat_idx * (fs/fs_nov));
     for i = beat_idx_fs
         sonified_beats(i:i+round(fs/20)) = rand(1,round(fs/20)+1); 
     end
     track_with_beats = x_t + sonified_beats;
     soundsc(track_with_beats,fs)
end