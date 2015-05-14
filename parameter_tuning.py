# parameter_tuning.py: methods used to tune the parameters of our
# beat tracking and first downbeat systems 
__author__ = "Nathan Lehrer"

import os
import core
import numpy as np
import mir_eval

def sonify_beats(x_t,fs,beat_times):
    # Add some clicks to better hear beat tracking
    
    # Parameters:
    # x_t: samples of audio file
    # fs: sample rate of file
    # beat_times: times of beats in seconds
    
    # Returns:
    # audio_with_beats: samples of audio file with clicks on estimated beats

    beat_sig = mir_eval.sonify.clicks(beat_times,fs,length=len(x_t))
    audio_with_beats = 0.5*x_t+beat_sig
    return audio_with_beats

def tempo_cost_all(x_t_array,fs,win_size_nov,hop_size_nov,win_size_tpo,hop_size_tpo,
				   min_bpm,max_bpm,actual_tempo_array,tolerance):
	# Compute the percent of tracks in a set that have greater tempo error than
	# a given tolerance where tempo error is the error in bpm between the estimated
	# and ground truth tempos

	# Parameters:
	# x_t_array: array of audio samples, where x_t_array[0] holds all
    # samples of the first track, and so on
	# fs: sample rate of audio files
	# win_size_nov, hop_size_nov, win_size_tpo, hop_size_tpo, min_bpm, max_bpm: 
	# parameters for get_beats
	# actual_tempo_array: array of ground truth tempos

	# Returns:
	# cost: decimal percent of tracks in a set that have greater than "tolerance" tempo error

    cost_accum = 0.
    for i in range(len(actual_tempo_array)):
		tempo = core.get_beats(x_t_array[i],fs,win_size_nov,hop_size_nov,win_size_tpo,
						hop_size_tpo,min_bpm,max_bpm)[2]
		tempo_error = abs(actual_tempo_array[i]-tempo)
		print actual_tempo_array[i],tempo		
		if (tempo_error > tolerance):
			cost_accum = cost_accum + 1
    cost = cost_accum/len(x_t_array)
    return cost

def first_downbeat_cost_all(x_t_array,fs,low_freq,high_freq,hop_sec,rms_thresh,offset,
							actual_first_downbeat_array):
	# Compute the percent of tracks in a set that have greater first downbeat error than
	# a given tolerance, where first downbeat error is the error in seconds between the estimated
	# and ground truth tempos

	# Parameters:
	# x_t_array: array of audio samples, where x_t_array[0] holds all
    # samples of the first track, and so on
	# fs: sample rate of audio files
	# low_freq, high_freq, hop_sec, rms_thresh, offset: 
	# parameters for get_first_downbeat
	# actual_tempo_array: array of ground truth tempos

	# Returns:
	# cost: decimal percent of tracks in a set that have greater than "tolerance" first downbeat error

	cost_accum = 0.;
	for i in range(len(x_t_array)):
		first_downbeat = core.get_first_downbeat(x_t_array[i],fs,low_freq,high_freq,hop_sec,
											rms_thresh,offset)[1]
		if (abs(first_downbeat-actual_first_downbeat_array[i]) > 0.5): # within 0.5 seconds
		    cost_accum = cost_accum + 1
		print actual_first_downbeat_array[i],first_downbeat
	cost = cost_accum/len(x_t_array)
	return cost


def histogram_segment_lengths(jamspath,track_names,x_t_array,fs,win_size_nov,hop_size_nov,win_size_tpo, 
		   					  hop_size_tpo,min_bpm,max_bpm,low_freq,high_freq,hop_sec,rms_thresh,offset,
							  fix_or_not,tol,beat_or_meas,ref_or_est):
	# From a set of (estimated or ground truth) segment boundary annotations and a set of tracks, 
	# compute two histograms: a histogram of the lengths of all segments in all tracks, in beats,
	# as computed by our PLP beat tracker, and a histogram of the placement of each segment,
	# in beats or measures, relative to the first downbeat

	# Parameters:
	# jamspath: folder in which the .jams boundary annotations or estimations reside
	# track_names: names of the tracks, including audio extensions
	# x_t_array: array of audio samples, where x_t_array[0] holds all
    # samples of the first track, and so on
	# fs: sample rate of audio files
	# win_size_nov, hop_size_nov, win_size_tpo, hop_size_tpo, min_bpm, max_bpm: 
	# parameters for get_beats
	# low_freq, high_freq, hop_sec, rms_thresh, offset: 
	# parameters for get_first_downbeat
	# fix_or_not: boolean, 1 = perform length fixing on the segment lengths, 0 = don't
	# tol: tolerance for length fixing, in beats/measures
	# beats_or_meas: boolean, 1 = histogram lengths/placements in beats, 0 = in measures
	# ref_or_est: boolean, 1 = annotated references, 0 = MSAF estimations (the jams files are different)

	# Returns:
	# length_histogram: histogram of segment lengths in beats/measures, where
	# length_histogram[n] = number of n-long segments
	# placement_histogram: histogram of segment boundary placements in beats/measures, where
	# placement_histogram[n] = number of boundaries falling on nth beat/meas after first downbeat

    length_histogram = np.zeros(400)
    placement_histogram = np.zeros(2000)
    for i in range(len(track_names)):
		# read .jams file to obtain estimated times
		print track_names[i]
		if track_names[i][-4:] == 'flac':
			f = open(jamspath+track_names[i][0:-4]+'jams','r')
		else:
			f = open(jamspath+track_names[i][0:-3]+'jams','r')
		lines = f.readlines()
		f.close()
		est_times = []
		if ref_or_est:		
			val_line = 9
			while (lines[val_line][13:18] == 'value'):
				est_times.append(lines[val_line][22:27])
				val_line += 12
		else:
			val_line = 43
			while (lines[val_line][13:18] == 'value'):
				est_times.append(lines[val_line][21:27])
				val_line += 15
			val_line -= 11
			est_times.append(lines[val_line][21:27])
		est_times_string = np.array(est_times, dtype='|S6')
		print est_times_string
		est_times = est_times_string.astype(np.float)

		# compute placements and lengths of segments	

		beats_idx,beat_times,tempo = core.get_beats(x_t_array[i],fs,win_size_nov,hop_size_nov,
				                               win_size_tpo,hop_size_tpo,min_bpm,max_bpm)
		first_downbeat_idx, first_downbeat_time = core.get_first_downbeat(x_t_array[i],fs,low_freq,high_freq,
				                                                     hop_sec,rms_thresh,offset)
		if not (beat_or_meas):
			beats_idx, beat_times = core.get_downbeats(onset_idx,first_downbeat_idx,fs,tempo)

		placements,lengths = core.compute_boundary_placements(est_times[0:-2],beat_times,first_downbeat_time)
		
		# compute histograms

		for l in lengths:
			length_histogram[l] += 1
		for p in placements:
			placement_histogram[p] += 1
    return length_histogram, placement_histogram
