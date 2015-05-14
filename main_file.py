__author__ = "Nathan Lehrer"

import core
import parameter_tuning as param
import librosa
import numpy as np
import matplotlib.pyplot as plt
	
if __name__ == '__main__':
	audiopath = '/home/nathan/documents/nyuclasses/mir/project/edm_collection/audio/'
	refpath = '/home/nathan/documents/nyuclasses/mir/project/edm_collection/references/'
	estpath = '/home/nathan/documents/nyuclasses/mir/project/edm_collection/estimations/'	
	track_names = ['1AfterDark.mp3','1Arcadia.mp3','1BigWednesday.mp3','1CelestialAnnihilation.mp3',
		  '1ChemicalBeats.mp3','1Clear.mp3','1CrossedOut.mp3','1EmbracingTheFuture.mp3',
		  '1EpicLastSong.mp3','1Evolution.mp3','1ForReal.mp3','1One.mp3','1Peace.mp3',
		  '1Satellites.wav','1Soul2000.mp3','1Technology.mp3','1TheGroove.mp3',
		  '1TheSeaFront.mp3','1TimeIsFire.mp3','1Transducer.mp3',
		  '4.mp3','AdagioForStrings.mp3','Bangarang.mp3','Breathe.mp3','Carcola.mp3',
		  'CarnMarth.mp3','Collide.mp3','&Down.mp3','FunkyShit.mp3','GodIsADJ.mp3',
		  'ICanSee.mp3','Insomnia.mp3','LavaLava.mp3','LittleBadGirl.mp3',
		  'NewYorkNewYork.mp3','Porcelain.mp3','Railing.mp3','RedAlert.mp3',
		  'RightIn.mp3','RightOnTime.mp3','SexyBoy.mp3','ShareTheFall.mp3',
		  'SmackMyBitchUp.mp3','Springer.mp3','Sweat.mp3','TheNumberSong.mp3',
		  'TheRockafellerSkank.mp3','TikTikTikTakTakTak.mp3','Titanium.mp3',
		  'TurnMeOn.mp3']
	actual_tempo_array = [170,128.3,115,120,121,126.7,136,130,154,151,125,128,172,170,175,167,
		         188,140,140,125,
		         167,140,110,130,128,167,127,123,125,133,116,127,125,127,130,
		         180,170,127,140,140,112,170,137,127,130,104,153,125,127,128]
	actual_first_downbeat_array = [57.554,30.133,33.038,8.56,0.054,4.683,0.153,14.83,25.036,44.552,
		                  0.172,7.791,0.009,49.996,65.894,69.223,10.324,0.119,29.195,31.511,
		                  0.238, 0.322, 9.095, 15.124, 0.127, 1.908, 1.052, 2.235,
		                  2.635, 7.661, 0.197, 62.400, 31.499, 7.708, 4.000, 10.473,
		                  1.095, 8.200, 0.368, 14.290, 0.476, 46.350, 49.795, 0.252,
		                  0.470, 0.307, 39.300, 0.142, 15.300, 15.400]
	
	# load audio samples
	#x_t_array,fs = core.load_x_t_array(audiopath,track_names)

	# get_beats default params
	win_size_nov = 1024
	hop_size_nov = 512
	win_size_tpo = 2048
	hop_size_tpo = 512
	min_bpm = 100
	max_bpm = 200

	# get_first_downbeat default params
	low_freq = 20.
	high_freq = 80.
	rms_thresh = 0.1
	hop_sec = 0.01
	offset = 0

# TEMPO OPTIMIZATION
	
	win_size_nov = 2048
	hop_size_nov = 1024

	# BASIC METHOD FOR OBTAINING RESULTS: (modified for each set of results)

	tolerance = 3 # within 3 bpm, any errors are probably due to a slightly wrong annotation 
	#results = []
	#for i in range(1,33):
	#	win_size_tpo = 256*i
	#	hop_size_tpo = win_size_tpo/2
	#	cost_all = param.tempo_cost_all(x_t_array,fs,win_size_nov,hop_size_nov,win_size_tpo,hop_size_tpo,
	#			   min_bpm,max_bpm,actual_tempo_array,tolerance)
	#	results.append(cost_all)
		
	# RESULTS:
	
	# Varying novelty window and hop size: 

	win_size_nov_array = [256,512,768,1024,1280,1536,1792,2048,2304,2560,2816,3072,3328,3584,3840,4096]

	# condition 1: win_size_tpo = 4096, hop_size_tpo = 1024, hop_size_nov = win_size_nov / 2
	novresults1 = [0.22, 0.24, 0.08, 0.04, 0.1, 0.06, 0.1, 0.12, 0.1, 0.1, 0.12, 0.1, 0.08, 0.1, 0.1, 0.1]
	
	# condition 2: win_size_tpo = 2048, hop_size_tpo = 512, hop_size_nov = win_size_nov / 2
	novresults2 = [0.24, 0.22, 0.06, 0.08, 0.06, 0.12, 0.06, 0.12, 0.1, 0.12, 0.14, 0.1, 0.08, 0.12, 0.1, 0.1]
	
	# condition 3: win_size_tpo = 4096, hop_size_tpo = 1024, hop_size_nov = win_size_nov / 4
	novresults3 = [0.26, 0.22, 0.08, 0.06, 0.06, 0.12, 0.06, 0.1, 0.12, 0.1, 0.14, 0.1, 0.1, 0.14, 0.12, 0.12]
	
	# condition 3: win_size_tpo = 2048, hop_size_tpo = 512, hop_size_nov = win_size_nov / 4	
	novresults4 = [0.62, 0.16, 0.14, 0.08, 0.08, 0.08, 0.1, 0.12, 0.14, 0.12, 0.12, 0.12, 0.14, 0.12, 0.1, 0.12]
	
	# Varying tempogram window and hop size:

	win_size_tpo_array = [ 256,512,768,1024,1280,1536,1792,2048,2304,2560,2816,
       					   3072,3328,3584,3840,4096,4352,4608,4864,5120,5376,5632,
       					   5888,6144,6400,6656,6912,7168,7424,7680,7936,8192]

	# condition 1: win_size_nov = 1024, hop_size_nov = 512, hop_size_tpo = win_size_tpo / 2
	tporesults1 = [0.58, 0.08, 0.1, 0.1, 0.06, 0.06, 0.08, 0.08, 0.04, 0.06, 0.04, 0.08, 0.06, 0.06, 0.06, 0.04, 0.04, 0.08, 0.04, 0.06, 0.08, 0.04, 0.06, 0.04, 0.04, 0.04, 0.04, 0.04, 0.08, 0.08, 0.04, 0.06]

	# condition 2: win_size_nov = 2048, hop_size_nov = 1024, hop_size_tpo = win_size_tpo / 2
	tporesults2 = [0.16, 0.16, 0.1, 0.1, 0.12, 0.12, 0.08, 0.12, 0.14, 0.12, 0.1, 0.08, 0.1, 0.1, 0.1, 0.1, 0.14, 0.1, 0.14, 0.14, 0.1, 0.1, 0.1, 0.1, 0.1, 0.08, 0.1, 0.14, 0.1, 0.1, 0.14, 0.12]
	
	# condition 3: win_size_nov = 1024, hop_size_nov = 512, hop_size_tpo = win_size_tpo / 4
	tporesults3 = [0.58, 0.08, 0.08, 0.12, 0.06, 0.06, 0.08, 0.08, 0.06, 0.04, 0.04, 0.08, 0.06, 0.04, 0.06, 0.04, 0.04, 0.06, 0.04, 0.08, 0.06, 0.04, 0.06, 0.04, 0.04, 0.04, 0.04, 0.04, 0.08, 0.06, 0.04, 0.06]
	
	# condition 4: win_size_nov = 2048, hop_size_nov = 1024, hop_size_tpo = win_size_tpo / 4	
	tporesults4 = [0.16, 0.16, 0.12, 0.1, 0.12, 0.12, 0.08, 0.12, 0.12, 0.12, 0.1, 0.08, 0.08, 0.1, 0.1, 0.12, 0.1, 0.12, 0.12, 0.12, 0.1, 0.1, 0.1, 0.12, 0.12, 0.1, 0.12, 0.12, 0.1, 0.1, 0.12, 0.14]	
	
	#plt.plot(win_size_nov_array,novresults1,label='condition 1')
	#plt.plot(win_size_nov_array,novresults2,label='condition 2')
	#plt.plot(win_size_nov_array,novresults3,label='condition 3')
	#plt.plot(win_size_nov_array,novresults4,label='condition 4')
	#plt.xlabel('Window size of novelty function')
	#plt.ylabel('Rate of tracks with mis-estimated tempo')
	#plt.legend()
	#plt.show()
	
	#plt.plot(win_size_tpo_array,tporesults1,label='condition 1')
	#plt.plot(win_size_tpo_array,tporesults2,label='condition 2')
	#plt.plot(win_size_tpo_array,tporesults3,label='condition 3')
	#plt.plot(win_size_tpo_array,tporesults4,label='condition 4')
	#plt.xlabel('Window size of tempogram function')
	#plt.ylabel('Rate of tracks with mis-estimated tempo')
	#plt.legend()
	#plt.show()

# FIRST DOWNBEAT OPTIMIZATION

	# Optimize rms_thresh and hop_sec
	#low_freq = 20.
	#high_freq = 80.
	#offset = 0.
	#rms_thresh_array_coarse = [0.01,0.03,0.1,0.3,1.]
	#hop_sec_array_coarse = [0.01,0.03,0.1,0.3,1.]
	#rms_thresh_array_fine = [0.08,0.09,0.1,0.11,0.12]
	#hop_sec_array_fine = [0.008,0.009,0.01,0.011,0.012]

	#results = np.zeros([len(rms_thresh_array),len(hop_sec_array)])
	#for i in range(len(rms_thresh_array)):
	#	for j in range(len(hop_sec_array)):
	#		print rms_thresh_array[i]
	#		print hop_sec_array[j]
	#		rms_thresh = rms_thresh_array[i]
	#		hop_sec = hop_sec_array[j]
	#		results[i][j] = param.first_downbeat_cost_all(x_t_array,fs,low_freq,high_freq,
	#												hop_sec,rms_thresh,offset,actual_first_downbeat_array)
	#results_coarse = [[ 0.62,  0.58,  0.52,  0.52,  0.56],
    #   [ 0.48,  0.44,  0.42,  0.3 ,  0.44],
    #   [ 0.08,  0.1 ,  0.14,  0.34,  0.58],
    #   [ 0.46,  0.6 ,  0.8 ,  0.9 ,  0.96],
    #   [ 1.  ,  1.  ,  1.  ,  1.  ,  1.  ]]	

	#results_fine = [[ 0.2 ,  0.14,  0.18,  0.14,  0.16],
    #   [ 0.12,  0.14,  0.12,  0.12,  0.12],
    #   [ 0.12,  0.12,  0.08,  0.08,  0.08],
    #   [ 0.1 ,  0.08,  0.1 ,  0.1 ,  0.06],
    #   [ 0.06,  0.06,  0.08,  0.06,  0.06]]

	# Optimize offset
	#rms_thresh = 0.12
	#hop_sec = 0.012
	
	#results = []
	#offset_array_coarse = [0.01,0.03,0.1,0.3,1.,3.]
	#for i in range(len(offset_array_coarse)):
	#	offset = offset_array_coarse[i]
	#	cost_all = 	param.first_downbeat_cost_all(x_t_array,fs,low_freq,high_freq,
	#												hop_sec,rms_thresh,offset,actual_first_downbeat_array)
	#	results.append(cost_all)

	# Optimize filter limits
	
	# Coarse
	#rms_thresh = 0.12
	#hop_sec = 0.012
	#low_freq_array_coarse = [10.,30.,50.,70.]
	#high_freq_array_coarse = [80.,100.,120.,140.,160.]
	#low_freq_array_fine = [10.,15.,20.,25.,30.]
	#high_freq_array_fine = [70.,75.,80.,85.,90.]

	#results = np.zeros([len(low_freq_array_fine),len(high_freq_array_fine)])
	#for i in range(len(low_freq_array_fine)):
	#	for j in range(len(high_freq_array_fine)):
	#		print low_freq_array_fine[i]
	#		print high_freq_array_fine[j]
	#		low_freq = low_freq_array_fine[i]
	#		high_freq = high_freq_array_fine[j]
	#		results[i][j] = param.first_downbeat_cost_all(x_t_array,fs,low_freq,high_freq,
	#												hop_sec,rms_thresh,offset,actual_first_downbeat_array)
	#results = [[ 0.06,  0.14,  0.22,  0.3 ,  0.3 ],
    #   [ 0.06,  0.12,  0.22,  0.28,  0.32],
    #   [ 0.14,  0.14,  0.22,  0.3 ,  0.32],
    #   [ 0.44,  0.16,  0.24,  0.28,  0.28]]

	#results = [[ 0.2 ,  0.14,  0.06,  0.06,  0.08],
    #   [ 0.18,  0.14,  0.06,  0.06,  0.1 ],
    #   [ 0.18,  0.12,  0.06,  0.06,  0.08],
    #   [ 0.18,  0.14,  0.06,  0.08,  0.08],
    #   [ 0.16,  0.16,  0.06,  0.08,  0.08]]

# RESOLVING AMBIGUITY WITH HISTOGRAMS

	win_size_nov = 1024
	hop_size_nov = 512
	rms_thresh = 0.12
	hop_sec = 0.012
	offset = 0
	low_freq = 20.
	high_freq = 80.
	win_size_tpo_array = [2304,2304,2560,2816,3072,3328,3584,3840,4096,4352,4608,4864,5120,5376,5632,
       					   5888,6144,6400,6656,6912,7168,7424,7680,7936,8192]
	jamspath = refpath	
	fix_or_not = 0 #no length fixing
	tol = 0 #for length fixing, doesn't matter
	beat_or_meas = 1 #measure segment lengths in beats
	ref_or_est = 1 #using annotated references

	results = []
	#for size in win_size_tpo_array:
	#	win_size_tpo = size
	#	hop_size_tpo = win_size_tpo/2
	#	length_histogram,placement_histogram = param.histogram_segment_lengths(jamspath,track_names,x_t_array,
	#						  fs,win_size_nov,hop_size_nov,win_size_tpo, hop_size_tpo,min_bpm,max_bpm,low_freq,
   	#						  high_freq,hop_sec,rms_thresh,offset,fix_or_not,tol,beat_or_meas,ref_or_est)
	#	score_all = length_histogram[32]+length_histogram[64]
	#	print score_all
	#	results.append(score_all)

	#results = [184.0, 184.0, 199.0, 193.0, 182.0, 200.0, 190.0, 193.0, 198.0, 197.0, 195.0, 192.0, 182.0, 179.0, 195.0, 175.0, 206.0, 193.0, 197.0, 206.0, 199.0, 208.0, 186.0, 196.0, 210.0]

	#plt.plot(win_size_tpo_array,results)
	#plt.xlabel('Window size of tempogram function')
	#plt.ylabel('Number of segments in the dataset with length 32 or 64 beats')
	#plt.show()

	# demo optimal parameters
	win_size_tpo = 8192
	hop_size_tpo = win_size_tpo/2
	beat_or_meas = 0
	length_histogram,placement_histogram = param.histogram_segment_lengths(jamspath,track_names,x_t_array,
							  fs,win_size_nov,hop_size_nov,win_size_tpo, hop_size_tpo,min_bpm,max_bpm,low_freq,
   							  high_freq,hop_sec,rms_thresh,offset,fix_or_not,tol,beat_or_meas,ref_or_est)
	plt.plot(length_histogram[0:100])
	plt.xlabel('Number of measures in a segment')
	plt.ylabel('Number of segments with that many measures')

	plt.figure()
	plt.plot(placement_histogram[0:500])
	plt.xlabel('Measure placement of a boundary relative to first downbeat')
	plt.ylabel('Number of boundaries with that placement')
	plt.show()
	
	
# MEASURE-SYNC DEMO	

	#win_size_tpo = 8192
	#beat_or_meas = 0
	#length_histogram,placement_histogram = param.histogram_segment_lengths(jamspath,track_names,x_t_array,
	#						  fs,win_size_nov,hop_size_nov,win_size_tpo, hop_size_tpo,min_bpm,max_bpm,low_freq,
   	#						  high_freq,hop_sec,rms_thresh,offset,fix_or_not,tol,beat_or_meas,ref_or_est)

