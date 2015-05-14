# core.py: core methods for our segmentation project
# including methods for beat tracking and first downbeat detection

__author__ = "Nathan Lehrer"

import os
import numpy as np
from numpy import mean, sqrt, square
import scipy.signal as scisig
import scipy.stats as scistat
import librosa
import matplotlib.pyplot as plt

def load_x_t_array(audiopath,track_names):
    # Load all 50 audio files into an array  

    # Parameters: 
    # audiopath: the path to the folder of audio files
    # track_names: an array of strings with the name and
    # extension of each audio file

    # Returns:
    # x_t_array: an array of the audio file values. 
    # x_t_array[0] holds all of the samples of the first track, and so on
    # fs: sample rate of the audio files (assumes all fs are equal)
    
    x_t_array = []
    for name in track_names:
        x_t,fs = librosa.load(audiopath+name)
        print name
        x_t_array.append(x_t)
    return x_t_array,fs    

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

def get_beats_first_downbeat_on(beat_times,first_downbeat_time): 
    # Get times of beats starting with first downbeat
    
    # Parameters:
    # beat_times: estimated times of beats
    # first_downbeat_time: estimated time of downbeat
    
    # Returns:
    # beat_times_dbo: beat times starting with the first downbeat
    
    # find beat index closest to detected first downbeat
    first_index = np.argmin(np.abs(np.array(beat_times)-first_downbeat_time))
    beat_times_fdbo = beat_times[first_index:]
    return beat_times_fdbo 

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

def compute_boundary_placements(boundary_times,beat_times,first_downbeat_time):
    # From a list of times of segment boundaries, compute the placements of those boundaries
    # with respect to the estimated first downbeat, as well as the length of each segment in beats
    
    # Parameters:
    # x_t: audio samples
    # fs: sample rate of audio
    # boundary_times: estimated or annotated times of a segment boundary, in seconds
    # beats_idx: indices of beats in samples. could also be indices of downbeats, in which case
    # this method computes the placements of boundaries in terms of measures
    # first_downbeat_idx: index of first downbeat in samples
    
    # Returns:
    # boundary_placements: placements of boundaries, in number of beats after first downbeat,
    # or measures after first downbeat depending on the input
    # segment_lengths: length in beats (or measures) of each segment from boundary to boundary

    beats_dbo = get_beats_first_downbeat_on(beat_times,first_downbeat_time)
    boundary_placements = []
    segment_lengths = []
    for i in range(len(boundary_times)):
        placement = np.argmin(np.abs(beats_dbo-boundary_times[i]))
        boundary_placements.append(placement)
        if (i > 0):
            segment_lengths.append(placement-boundary_placements[i-1])
    return boundary_placements,segment_lengths

