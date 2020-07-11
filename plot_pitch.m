% Nathan Lehrer N17119112
function [] = plot_pitch(filepath, win_size, hop_size, min_lag, max_lag)
    % Compute and plot pitch estimates for unbiased autocorrelation function and for yin.

    % Parameters
    % ----------
    % filepath : string
    %     path to .wav file
    % win_size : int
    %     window size (in samples)
    % hop_size : int
    %     hop size (in samples)
    % min_lag : int
    %     minimum possible lag value (in samples)
    % max_lag : int
    %     maximum possible lag value (in samples)
    %     
    % Returns
    % -------
    % None

    [x_t, fs, t] = import_audio(filepath);
    [pitch_hz_acf, t_pitch_acf] = detect_pitch_acf(x_t, t, fs, win_size, ...
      hop_size, min_lag, max_lag);
    [pitch_hz_yin, t_pitch_yin] = detect_pitch_yin(x_t, t, fs, win_size, ...
      hop_size, min_lag, max_lag);
    pitch_midi_acf = 12*log2(pitch_hz_acf/440) + 69; %convert to midi
    pitch_midi_yin = 12*log2(pitch_hz_yin/440) + 69; %convert to midi
    subplot(1,2,1);
    plot(t_pitch_acf,pitch_midi_acf,'k.');
    title('Pitches Using Spectral ACF')
    xlabel('Time, sec')
    ylabel('Pitch, MIDI value')
    subplot(1,2,2);
    plot(t_pitch_yin,pitch_midi_yin,'k.');
    title('Pitches Using Yin')
    xlabel('Time, sec')
end