"""
MSAF module to extract the audio features using librosa.

Original code: Oriol Nieto
Modified by: Nathan Lehrer to include methods from EDM segmentation project

Features to be computed:

- MFCC: Mel Frequency Cepstral Coefficients
- HPCP: Harmonic Pithc Class Profile
- Beats
"""

import datetime
import librosa
from joblib import Parallel, delayed
import logging
import numpy as np
import os
import json
import scipy.signal as scisig
import scipy.stats as scistat
from numpy import mean, sqrt, square

# Local stuff
import msaf
from msaf import jams2
from msaf import utils
from msaf import input_output as io
from msaf.input_output import FileStruct


def compute_beats(y_percussive, sr=22050):
	"""Computes the beats using librosa.

	Parameters
	----------
	y_percussive: np.array
		Percussive part of the audio signal in samples.
	sr: int
		Sample rate.

	Returns
	-------
	beats_idx: np.array
		Indeces in frames of the estimated beats.
	beats_times: np.array
		Time of the estimated beats.
	"""
	logging.info("Estimating Beats...")
	min_bpm = 100
	max_bpm = 200
	win_size_nov = 1024
	hop_size_nov = win_size_nov/2
	win_size_tpo = 8192
	hop_size_tpo = win_size_tpo/2
	beats_idx_prelim, beat_times, tempo = get_beats(y_percussive,sr,win_size_nov,hop_size_nov,
												win_size_tpo,hop_size_tpo,min_bpm,max_bpm)
	print beat_times[0:20]
	low_freq = 20.
	high_freq = 80.
	rms_thresh = 0.12
	hop_sec = 0.012
	offset = 0
	first_downbeat_idx = get_first_downbeat(y_percussive,sr,low_freq,high_freq,hop_sec,rms_thresh,offset)[0]
	meas_times = get_downbeats(beats_idx_prelim,first_downbeat_idx,sr,tempo)[1]
	beats_idx = librosa.core.time_to_frames(beat_times, sr, hop_length=msaf.Anal.hop_size)

	return np.array(beats_idx), np.array(beat_times)


def compute_features(audio, y_harmonic):
    """Computes the HPCP and MFCC features.

    Parameters
    ----------
    audio: np.array(N)
        Audio samples of the given input.
    y_harmonic: np.array(N)
        Harmonic part of the audio signal, in samples.

    Returns
    -------
    mfcc: np.array(N, msaf.Anal.mfcc_coeff)
        Mel-frequency Cepstral Coefficients.
    hpcp: np.array(N, 12)
        Pitch Class Profiles.
    tonnetz: np.array(N, 6)
        Tonal Centroid features.
    """
    logging.info("Computing Spectrogram...")
    S = librosa.feature.melspectrogram(audio,
                                       sr=22050,
                                       n_fft=msaf.Anal.frame_size,
                                       hop_length=msaf.Anal.hop_size,
                                       n_mels=msaf.Anal.n_mels)

    logging.info("Computing MFCCs...")
    log_S = librosa.logamplitude(S, ref_power=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=msaf.Anal.mfcc_coeff).T

    logging.info("Computing HPCPs...")
    hpcp = librosa.feature.chroma_cqt(y=y_harmonic,
                                      sr=22050,
                                      hop_length=msaf.Anal.hop_size).T

    #plt.imshow(hpcp.T, interpolation="nearest", aspect="auto"); plt.show()
    logging.info("Computing Tonnetz...")
    tonnetz = utils.chroma_to_tonnetz(hpcp)
    return mfcc, hpcp, tonnetz


def save_features(out_file, features):
    """Saves the features into the specified file using the JSON format.

    Parameters
    ----------
    out_file: str
        Path to the output file to be saved.
    features: dict
        Dictionary containing the features.
    """
    logging.info("Saving the JSON file in %s" % out_file)
    out_json = {"metadata": {"version": {"librosa": librosa.__version__}}}
    out_json["analysis"] = {
        "dur": features["anal"]["dur"],
        "frame_rate": msaf.Anal.frame_size,
        "hop_size": msaf.Anal.hop_size,
        "mfcc_coeff": msaf.Anal.mfcc_coeff,
        "n_mels": msaf.Anal.n_mels,
        "sample_rate": msaf.Anal.sample_rate,
        "window_type": msaf.Anal.window_type
    }
    out_json["beats"] = {
        "times": features["beats"].tolist()
    }
    out_json["timestamp"] = \
        datetime.datetime.today().strftime("%Y/%m/%d %H:%M:%S")
    out_json["framesync"] = {
        "mfcc": features["mfcc"].tolist(),
        "hpcp": features["hpcp"].tolist(),
        "tonnetz": features["tonnetz"].tolist()
    }
    out_json["est_beatsync"] = {
        "mfcc": features["bs_mfcc"].tolist(),
        "hpcp": features["bs_hpcp"].tolist(),
        "tonnetz": features["bs_tonnetz"].tolist()
    }
    try:
        out_json["ann_beatsync"] = {
            "mfcc": features["ann_mfcc"].tolist(),
            "hpcp": features["ann_hpcp"].tolist(),
            "tonnetz": features["ann_tonnetz"].tolist()
        }
    except:
        logging.warning("No annotated beats")

    # Actual save
    with open(out_file, "w") as f:
        json.dump(out_json, f, indent=2)

def novelty_sf(x_t,win_size,hop_size):
    # Compute novelty function using spectral flux
    
    # Parameters:
    # x_t: audio samples
    # win_size, hop_size: the window and hop size of the spectrogram
    
    # Returns:
    # n_t_sf: the spectral flux novelty function of the track

	X_spec = librosa.core.stft(x_t, n_fft=win_size,hop_length=hop_size, 
		                       win_length=win_size)
	X_mag_spec = np.abs(X_spec) 
	# spec flux is frames-1 long since no useful info about deriv of frame 1
	X_mag_spec_flux = X_mag_spec[:,1:] - X_mag_spec[:,0:-1]
	n_t_sf = (2./win_size) * sum(np.square(np.divide(X_mag_spec_flux + 
		                                             abs(X_mag_spec_flux),2.)))
	return n_t_sf

def get_tempogram(x_t,fs,win_size_nov,hop_size_nov,win_size_tpo,hop_size_tpo,min_bpm,max_bpm):
    # Compute tempogram as a spectrogram of the novelty function
    
    # Parameters:
    # win_size_nov, hop_size_nov: the window and hop size of of the novelty function
    # win_size_tpo, hop_size_tpo: the window and hop size of the tempogram
    # min_bpm, max_bpm: the minimum and maximum possible tempo in beats per minute
    
    # Returns:
    # tempogram: the tempogram of the track; the nth column is the complex fft of
    # the nth value in n_t
    # min_bin: the index of the lowest fft bin used (corresponds to min_bpm)
    # n_t_sf: the spectral flux novelty function of the track
    # fs_nov: the sample rate of the novelty function, in Hz 
    
    # For novelty function, use half the hop size, but for fs_nov,
    # pretend the hop size was hop_size_nov. See report for details.
	 
	#n_t_sf = librosa.onset.onset_strength(x_t,fs,hop_length=hop_size_nov/2)   
	n_t_sf = novelty_sf(x_t,win_size_nov,hop_size_nov/2) 
	fs_nov = float(fs)/hop_size_nov
	tempogram_full = librosa.core.stft(n_t_sf, n_fft=win_size_tpo, 
		                               hop_length=hop_size_tpo, win_length=win_size_tpo)
	min_bpm_hz = min_bpm/60.
	max_bpm_hz = max_bpm/60.
	min_bin = round((min_bpm_hz/fs_nov)*win_size_tpo)
	max_bin = round((max_bpm_hz/fs_nov)*win_size_tpo)
	tempogram = tempogram_full[min_bin:max_bin]
	return tempogram, min_bin, n_t_sf, fs_nov

def max_tempogram(tempogram, min_bin, fs_nov, win_size_tpo):
    # Compute optimal tempo values over time, and corresponding phases
    
    # Parameters:
    # tempogram: the tempogram of a track
    # min_bin: the index of the lowest fft bin used
    # fs_nov: the sample rate of the novelty function, in Hz
    # win_size_tpo: the window size of the tempogram
    
    # Returns:
    # omega: optimal tempo value in each tempogram window
    # phi: phase value in each window corresponding to optimal tempo value
    
    ind = np.argmax(abs(tempogram),0)
    omega = np.multiply((np.add(min_bin,ind)),(fs_nov/win_size_tpo)*60)
    phi = []
    for i in range(len(tempogram[1])):
        phi.append( (1/(2*np.pi))*np.angle(tempogram[ind[i]][i]) )
    return omega, phi

def compute_plp(omega,phi,n_t,win_size_tpo,hop_size_tpo,fs_nov):
    # Compute plp function using an overlap-add of cosine kernels
    
    # Parameters: 
    # omega: optimal tempo value in each tempogram window
    # phi: phase value in each window corresponding to optimal tempo value
    # n_t: a novelty function
    # win_size_tpo, hop_size: the window and hop size of the tempogram
    # fs_nov: the sample rate of the novelty function, in Hz
    
    # Returns:
    # plp: PLP function (Grosche, 09) that can be peak picked for beat times
    
    NT = len(omega)
    NNT = len(n_t)
    windows = np.zeros((NT,NNT))
    cosines = np.zeros((NT,NNT))
    kernels = np.zeros((NT,NNT))
    omega_hz = np.divide(omega,60.)
    for i in range(NT-1):
        length = len(windows[i][i*hop_size_tpo : i*hop_size_tpo + win_size_tpo])
        windows[i][i*hop_size_tpo : i*hop_size_tpo + win_size_tpo] = np.hamming(length)
        local_time = np.divide(range(length),fs_nov)
        cosines[i][i*hop_size_tpo : i*hop_size_tpo + 
                   win_size_tpo] = np.cos(np.multiply(2*np.pi,(np.subtract
                                    (np.multiply(omega_hz[i],local_time),phi[i]))))
        kernels[i] = np.multiply(windows[i], cosines[i])
    plp = sum(kernels)
    plp[plp<0]=0
    return plp
    
def beats_from_plp(plp,fs_nov):
    # Pick peak a plp function to find beat times
    
    # Parameters:
    # plp: PLP function
    # fs_nov: sample rate of the novelty function (= sample rate of plp function)
    
    # Returns:
    # beat_times: times of estimated beats in track, in seconds
    
    beats_idx = []
    # Only a naive pick peaking method is needed here,
    # because these are perfect cosines
    for i in range(1,len(plp)-1):
        if (plp[i] > plp[i-1]) and (plp[i] > plp[i+1]):
            beats_idx.append(i)
    beat_times = np.array(beats_idx)/fs_nov
    return beat_times

def get_beats(x_t,fs,win_size_nov,hop_size_nov,win_size_tpo,hop_size_tpo,min_bpm,max_bpm):
    # Using the plp method, extract the beat times from an audio file
    
    # Parameters:
    # x_t: samples of audio file
    # fs: sample rate of file
    # win_size_nov, hop_size_nov: the window and hop size of of the novelty function
    # win_size_tpo, hop_size_tpo: the window and hop size of the tempogram
    # min_bpm, max_bpm: the minimum and maximum possible tempo, in bpm
    
    # Returns:
    # beats_idx: indices of beats in samples
    # beat_times: times of beats in seconds
    # tempo: most frequent tempo in the track, in bpm
    
    tempogram, min_bin, n_t, fs_nov = get_tempogram(x_t,fs,win_size_nov,hop_size_nov,
                                                    win_size_tpo,hop_size_tpo,min_bpm,max_bpm)
    omega,phi = max_tempogram(tempogram, min_bin, fs_nov, win_size_tpo)
    plp = compute_plp(omega,phi,n_t,win_size_tpo,hop_size_tpo,fs_nov)
    beat_times = beats_from_plp(plp,fs_nov)
    beats_idx = np.round(np.multiply(beat_times,fs))
    # Calculate tempo as the most frequently occurring tempo in omega
    tempo = scistat.mode(omega)[0] 
    return beats_idx, beat_times, tempo

def get_first_downbeat(x_t,fs,low_freq,high_freq,hop_sec,rms_thresh,offset):
    # Estimate the time of the first bass-heavy downbeat in a track
    
    # Parameters: 
    # x_t: samples of audio file
    # fs: sample rate of file
    # low_freq, high_freq: boundaries in Hz of filter to capture kick drum
    # or other bassy instrument
    # hop_sec: the window/hop size of the nonoverlapping windows over which
    # rms is calculated
    # rms_thresh: energy threshold over which a nonoverlapping window is 
    # considered to have the first downbeat
    # offset: offset for peak picking the novelty function over the chosen window
    
    # Returns:
    # first_downbeat_idx: index in samples of first downbeat
    # first_downbeat_time: time in seconds of first downbeat
    
	b, a = scisig.butter(1, np.divide([low_freq, high_freq],fs/2), btype='bandpass')
	x_t_filt = scisig.filtfilt(b,a,x_t)
	rms_hop = hop_sec*fs #samples
	rms_too_low = 1
	position = 0
	track_length = len(x_t)
	while(rms_too_low and (position < track_length)):
		x_t_portion = x_t_filt[position:position+rms_hop]
		if (sqrt(mean(square(x_t_portion))) > rms_thresh):
		    rms_too_low = 0;
		    break;
		position = position + rms_hop

	win_size_nov = 2048
	hop_size_nov = 64
	fs_nov = float(fs)/hop_size_nov

	n_t = novelty_sf(x_t_portion,win_size_nov,hop_size_nov)
	first_onset_frame = 0
	for i in range(1,len(n_t)-1):
	    if (n_t[i] > n_t[i-1]) and (n_t[i] > n_t[i+1]) and (n_t[i] > offset):
	        first_onset_frame = i
	        break;

	first_time = float(first_onset_frame)/fs_nov

	first_downbeat_idx = round(first_time*fs+position)
	first_downbeat_time = first_time+float(position)/fs
	return first_downbeat_idx, first_downbeat_time

def get_downbeats(beats_idx,first_downbeat_idx,fs,tempo):
    # Get times of all downbeats, assuming 4 beats per measure
    
    # Parameters:
    # beats_idx: estimated beat indices in samples
    # first_idx: estimated index in samples of first downbeat
    # fs: sample rate of file
    # tempo: estimated tempo from tempogram (assuming consistent tempo)
    
    # Returns:
    # downbeats_idx: indices in frames of estimated downbeats
    # downbeat_times: times in seconds of estimated downbeats
    
    first_index = np.argmin(np.abs(beats_idx-first_downbeat_idx))
    beats_idx_dbon = beats_idx[first_index:]
    downbeats_idx = [0]
    downbeat_times = [0]
    hop = (60./tempo)*4.*fs
    position = first_downbeat_idx
    while(position < beats_idx[-1]):
        new_index = np.argmin(np.abs(beats_idx-position))
        downbeats_idx.append(beats_idx[new_index])
        downbeat_times.append(float(beats_idx[new_index])/fs)
        position = position + hop
    return downbeats_idx, downbeat_times

def compute_beat_sync_features(features, beats_idx):
    """Given a dictionary of features, and the estimated index frames,
    calculate beat-synchronous features."""
    bs_mfcc = librosa.feature.sync(features["mfcc"].T, beats_idx, pad=False).T
    bs_hpcp = librosa.feature.sync(features["hpcp"].T, beats_idx, pad=False).T
    bs_tonnetz = librosa.feature.sync(features["tonnetz"].T, beats_idx, pad=False).T
    return bs_mfcc, bs_hpcp, bs_tonnetz


def compute_features_for_audio_file(audio_file):
    """
    Parameters
    ----------
    audio_file: str
        Path to the audio file.

    Returns
    -------
    features: dict
        Dictionary of audio features.
    """
    # Load Audio
    logging.info("Loading audio file %s" % os.path.basename(audio_file))
    audio, sr = librosa.load(audio_file, sr=22050)

    # Compute harmonic-percussive source separation
    logging.info("Computing Harmonic Percussive source separation...")
    y_harmonic, y_percussive = librosa.effects.hpss(audio)

    # Output features dict
    features = {}

    # Compute framesync features
    features["mfcc"], features["hpcp"], features["tonnetz"] = \
        compute_features(audio, y_harmonic)

    # Estimate Beats
    features["beats_idx"], features["beats"] = compute_beats(
        y_percussive, sr=22050)

    # Compute Beat-sync features
    features["bs_mfcc"], features["bs_hpcp"], features["bs_tonnetz"] = \
        compute_beat_sync_features(features, features["beats_idx"])

    # Analysis parameters
    features["anal"] = {}
    features["anal"]["frame_rate"] = msaf.Anal.frame_size
    features["anal"]["hop_size"] = msaf.Anal.hop_size
    features["anal"]["mfcc_coeff"] = msaf.Anal.mfcc_coeff
    features["anal"]["sample_rate"] = msaf.Anal.sample_rate
    features["anal"]["window_type"] = msaf.Anal.window_type
    features["anal"]["n_mels"] = msaf.Anal.n_mels
    features["anal"]["dur"] = audio.shape[0] / float(msaf.Anal.sample_rate)

    return features


def compute_all_features(file_struct, sonify_beats=False, overwrite=False,
                         out_beats="out_beats.wav"):
    """Computes all the features for a specific audio file and its respective
        human annotations. It creates an audio file with the sonified estimated
        beats if needed.

    Parameters
    ----------
    file_struct: FileStruct
        Object containing all the set of file paths of the input file.
    sonify_beats: bool
        Whether to sonify the beats.
    overwrite: bool
        Whether to overwrite previous features JSON file.
    out_beats: str
        Path to the new file containing the sonified beats.
    """

    # Output file
    out_file = file_struct.features_file

    if os.path.isfile(out_file) and not overwrite:
        return  # Do nothing, file already exist and we are not overwriting it

    # Compute the features for the given audio file
    features = compute_features_for_audio_file(file_struct.audio_file)

    # Save output as audio file
    if sonify_beats:
        logging.info("Sonifying beats...")
        fs = 44100
        audio, sr = librosa.load(file_struct.audio_file, sr=fs)
        msaf.utils.sonify_clicks(audio, features["beats"], out_beats, fs,
                                 offset=0.0)

    # Read annotations if they exist in path/references_dir/file.jams
    if os.path.isfile(file_struct.ref_file):
        jam = jams2.load(file_struct.ref_file)

        # If beat annotations exist, compute also annotated beatsync features
        if jam.beats != []:
            logging.info("Reading beat annotations from JAMS")
            annot = jam.beats[0]
            annot_beats = []
            for data in annot.data:
                annot_beats.append(data.time.value)
            annot_beats = np.unique(annot_beats)
            annot_beats_idx = librosa.time_to_frames(
                annot_beats, sr=22050,
                hop_length=msaf.Anal.hop_size)
            features["ann_mfcc"], features["ann_hpcp"], \
                features["ann_tonnetz"] = \
                compute_beat_sync_features(features, annot_beats_idx)

    # Save output as json file
    save_features(out_file, features)


def process(in_path, sonify_beats=False, n_jobs=1, overwrite=False,
            out_file="out.json", out_beats="out_beats.wav"):
    """Main process to compute features.

    Parameters
    ----------
    in_path: str
        Path to the file or dataset to compute the features.
    sonify_beats: bool
        Whether to sonify the beats on top of the audio file
        (single file mode only).
    n_jobs: int
        Number of threads (collection mode only).
    overwrite: bool
        Whether to overwrite the previously computed features.
    out_file: str
        Path to the output json file (single file mode only).
    out_beats: str
        Path to the new file containing the sonified beats.
    """

    # If in_path it's a file, we only compute one file
    if os.path.isfile(in_path):
        file_struct = FileStruct(in_path)
        file_struct.features_file = out_file
        compute_all_features(file_struct, sonify_beats, overwrite, out_beats)

    elif os.path.isdir(in_path):
        # Check that in_path exists
        utils.ensure_dir(in_path)

        # Get files
        file_structs = io.get_dataset_files(in_path)

        # Compute features using joblib
        Parallel(n_jobs=n_jobs)(delayed(compute_all_features)(
            file_struct, sonify_beats, overwrite, out_beats)
            for file_struct in file_structs)


