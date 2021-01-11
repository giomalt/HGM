import pylab
import cv2
import sys 
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit
import time
from datetime import datetime
import os

from pathlib import Path

import torch

import scipy.signal # conv 

## perform fitting in less time using more cores
from joblib import Parallel, delayed
import multiprocessing
####################################################

def data_channels_from_binary(scope_bin_file, ch_len):
    ''' Read oscilloscope raw data (4 Ch in binary Big Endian format) and return the 4 channel traces. '''
    with open(scope_bin_file, "rb") as f:
        data_bin_loaded = f.read()
    # 950 µs ± 14.9 µs per loop
    
    # remove 'header' chars from byte string and ignore last new line carriage string
    header = data_bin_loaded[0:8]
    data_no_header = data_bin_loaded.replace(header, b'')[:-1]

    data_int = []
    # convert string of bytes to array of integers
    for idx in range(0, ch_len*2*4, 2):
        data_int.extend([ int.from_bytes(data_no_header[idx:idx+2], byteorder='big', signed=True) ])
    
    data_int = np.array(data_int).astype(int)

    j=0; trig_seq = data_int[j*ch_len:(j+1)*ch_len]
    j=1; heterodyne = data_int[j*ch_len:(j+1)*ch_len]
    j=2; AOM = data_int[j*ch_len:(j+1)*ch_len]
    j=3; trig_frame = data_int[j*ch_len:(j+1)*ch_len]

    return trig_seq, heterodyne, AOM, trig_frame

def cosinusoid_with_bias(x, a, w, phi, b):
    return a*np.cos(w*x+phi) + b

def sequence_bounds_from_trig_frame(trig_frame):
    '''
    This function finds the beginning and the end of the sequence based on the frame trigger signal.
    DMD sequence should have dark time after the last element to make algorithm working.
    The algorithms is based on measuring the distanse between frame trigger edges.
    '''
    is_signal = ( (trig_frame > 0.5*(np.max(trig_frame) + np.min(trig_frame))))
    frame_change_arr = np.diff(is_signal,1)

    frame_change_points = np.where(frame_change_arr == 1)[0]
    dist_between_change_points = np.diff(frame_change_points, 1)
    dist_between_change_points_sorted = np.sort(dist_between_change_points)

    where_dist_between_change_points_jumps = np.where(np.diff(dist_between_change_points_sorted)>2)
    where_dist_between_frames = where_dist_between_change_points_jumps[0][0]
    dist_between_frames = dist_between_change_points_sorted[where_dist_between_frames+1]
    number_of_dark_times = len(np.where(dist_between_change_points_sorted > dist_between_frames*1.02)[0])

    '''If scope trig_frame starts and ends with dark time'''
    if number_of_dark_times == 0:
        seq_start = frame_change_points[0]
        seq_end = np.min([frame_change_points[-1] + dist_between_frames, len(trig_frame) - 1])

    else:
        where_dist_between_dark_times = where_dist_between_change_points_jumps[0][-1]
        min_dist_between_dark_times = dist_between_change_points_sorted[where_dist_between_dark_times+1]

        dark_time_positions =\
        frame_change_points[np.where(dist_between_change_points >= min_dist_between_dark_times )[0] + 1]

        '''If scope trig_frame only starts with dark time'''
        if (len(dark_time_positions) == 1)&(dark_time_positions[0]<len(trig_frame)/2):
            seq_start = dark_time_positions[0]
            seq_end = np.min([frame_change_points[-1] + dist_between_frames, len(trig_frame) - 1])
            '''If scope trig_frame only ends with dark time'''
        elif (len(dark_time_positions) == 1)&(dark_time_positions[0]>len(trig_frame)/2):
            seq_start = frame_change_points[0]
            seq_end = dark_time_positions[0] - np.max(dist_between_change_points) + dist_between_frames
            '''If scope trig_frame starts and ends with remainings of neighbour sequences'''
        else:
            seq_start = dark_time_positions[0]
            seq_end = dark_time_positions[-1] - np.max(dist_between_change_points) + dist_between_frames

    return np.int(seq_start), np.int(seq_end)
    
def frames_start_stop(trig_frame, n_frames = 400): ###################################
    ''' 
    From the oscilloscope trace of DMD frame trigger,
    return the position of DMD frames starts and stops.'''
    is_frame = (trig_frame < np.mean(trig_frame)).astype(int) 

    frames_start_bool = ( is_frame - np.roll(is_frame,1) ) == 1
    frames_stop_bool = ( is_frame - np.roll(is_frame,1) ) == -1

    number_of_obj = np.sum(frames_start_bool)    

    if not(number_of_obj)==n_frames:
        print('! I did not find {} DMD frames, but {} !'.format(n_frames, number_of_obj))

    frames_start = (np.where(frames_start_bool == True)[0])
    frames_stop = (np.where(frames_stop_bool == True)[0])[1:]
    # we need to add the stop point of last frame (DMD trig frame sent before each frame is displayed)
    frames_stop = np.append(frames_stop, frames_stop[-1]+100); 

    if not(len(frames_start))==len(frames_stop):
        print('     Detected {} frame starts, {} frames stop'.format(len(frames_start), len(frames_stop)))
    if not(len(frames_start))==n_frames:
        print('     Detected {} frame starts, {} frames stop'.format(len(frames_start), len(frames_stop)))                               

    return frames_start, frames_stop, len(frames_start)

def norm_data(array):    
    '''Added to return normalisation factor'''
    ac_signal = array - np.average(array)
    return ac_signal/np.max(ac_signal), np.max(ac_signal)
    
def moving_avg(data, n):
    '''
    Return the moving average of an input signal.
    Used to apply a low-pass filter on the signal before fitting it with a sinusoid.
    Args:
        data (arr) - a noisy signal
        n (int) - number of points over which we perform the moving average 
    '''
    data = data.reshape(data.shape[0])
    return np.convolve(data, np.ones((n,))/n, mode='valid')

def preprocess(
    exp_folder,
    total_samples,
    useful_frames,
    HG_basis,

    save_folder=None,
    
    scope_trace_points=55000,
    ch0_vert_scale = 20,
    ch1_vert_scale = 3,
    guess_fit_freq = 0.52*0.5,
    moving_avg_kernel = None,
    
    repetition = 1,
    
    parallel_processing = True, # set to false to plot data and fits
    debug = False,
    
    power_norm = True,
    load_fits = True
):

    samples_per_trace = get_samples_per_trace(total_samples, useful_frames)
            
    j = np.zeros((total_samples,HG_basis*HG_basis*2))
    j_and_phase_ref = np.zeros((total_samples,HG_basis*HG_basis*4)) # save both photocurrents and phase refs
    
    phase_ref_amplitude_reference = np.zeros((HG_basis*HG_basis)) # power norm using phase ref
    mode_LO_used_for_power_norm = 0 # mode (0,0)
    
    # ~ ~ ~  fit data if you have not done it previsouly ~ ~ ~ #
    if not(load_fits):
        fit_raw_data(samples_per_trace, HG_basis, exp_folder, save_folder, scope_trace_points, ch0_vert_scale, ch1_vert_scale, guess_fit_freq,
                    parallel_processing, debug)

    # ~ ~ ~  calculate j from fitted data ~ ~ ~ #
    current_sample = 0
    
    ## calculate average power of phase ref for all HG modes. Use this for normalization
    if power_norm:
        amp_reference_phase_ref = np.zeros((HG_basis*HG_basis,1))
        for seq_n, frames_per_sequence in enumerate(samples_per_trace):
            scope_amplitudes = torch.load(save_folder+'seq/'+str(seq_n)+'_amp.pt')
            scope_amplitudes = scope_amplitudes.reshape(HG_basis*HG_basis, frames_per_sequence)
            scope_amplitudes = scope_amplitudes.numpy()
            scope_amplitudes = np.nan_to_num(scope_amplitudes)
            
            amp_reference_phase_ref += scope_amplitudes[mode_LO_used_for_power_norm,0]#.reshape(HG_basis*HG_basis,1)    
            
        amp_reference_phase_ref = amp_reference_phase_ref / (seq_n+1)
#     plt.plot(amp_reference_phase_ref, '.-'); plt.show()


    for seq_n, frames_per_sequence in enumerate(samples_per_trace):
        phase_diff = torch.load(save_folder+'seq/'+str(seq_n)+'_phase.pt')
        scope_amplitudes = torch.load(save_folder+'seq/'+str(seq_n)+'_amp.pt')

        phase_diff = phase_diff.reshape(HG_basis*HG_basis, frames_per_sequence)
        scope_amplitudes = scope_amplitudes.reshape(HG_basis*HG_basis, frames_per_sequence)

        phase_diff = phase_diff.numpy()
        scope_amplitudes = scope_amplitudes.numpy()
        
        # substitute nan with zeros
        phase_diff = np.nan_to_num(phase_diff)
        scope_amplitudes = np.nan_to_num(scope_amplitudes)
        
        # for each sample, subtract the phase of 'phase_ref' frame
        phase_diff[:, :] -= phase_diff[:, 0].reshape(HG_basis*HG_basis, 1)
                        
        # calculate samples complex photocurrents
        j_n = scope_amplitudes[:,1:-1]*np.exp(1j* phase_diff[:,1:-1] )

        if power_norm:
#             if seq_n == 0: # use phase_ref in sequence 0 as amplitude reference in normalization
#                 amp_reference_phase_ref = scope_amplitudes[:,0]         
            amp_current_phase_ref = scope_amplitudes[mode_LO_used_for_power_norm,0]#.reshape(HG_basis*HG_basis,1) 
    
#             if phase ref power high enough, use it for power norm
            amp_norm = amp_reference_phase_ref / amp_current_phase_ref
            plt.plot(amp_norm, '.-'); plt.show()

            j_n = j_n * amp_norm

        
        j_n = np.transpose(j_n)

        j[current_sample:current_sample+frames_per_sequence-2,0::2] = np.real(j_n)
        j[current_sample:current_sample+frames_per_sequence-2,1::2] = np.imag(j_n)
        current_sample+=frames_per_sequence-2
        
    return j

    
def get_samples_per_trace(total_samples, useful_frames):
    '''
    Calculate the number of frames (samples, phase ref and alignment square) in each acquired oscilloscope trace.
    Args:
        total_samples (int) - total number of useful samples, i.e. 30000
        useful_frames (int) - total number of useful frames per oscilloscope trace, i.e. 398
        
    Returns:
        samples_per_trace (array) - indicates the number of useful samples in each oscilloscope trace, i.e. (398, 398, .. , 206)
    '''    
    number_of_sequences = np.int(np.ceil(total_samples / useful_frames))   
    frames_last_sequence = total_samples - (number_of_sequences-1)*useful_frames
    
    samples_per_trace = np.ones(( number_of_sequences )) * useful_frames
    samples_per_trace[-1] = frames_last_sequence
    samples_per_trace += 2
    
    return samples_per_trace.astype(int)


def fit_raw_data(samples_per_trace, HG_basis, exp_folder, save_folder, scope_trace_points, ch0_vert_scale, ch1_vert_scale, guess_fit_freq, 
                 parallel_processing, debug):
    '''
    Fit raw data of oscilloscope traces and save them as tensors.
    
    Save:
        phase_diff (matrix) - phase difference between CH0 or 1 (HD signal) and CH2 (AOM)
                              Matrix of dimension (HG_basis x HG_basis, frames_per_sequence)
        scope_amplitudes (matrix) - amplitude of fitted signal of CH0 or 1
                                    Matrix of dimension (HG_basis x HG_basis, frames_per_sequence)
    '''
    
    num_cores = multiprocessing.cpu_count() # parallel processing
    
    current_sample=0
    
    for seq_n, frames_per_sequence in enumerate(samples_per_trace):
        print('Fitting sequence: '+str(seq_n))
        phase_diff = np.zeros((HG_basis, HG_basis, frames_per_sequence))
        scope_amplitudes = np.zeros((HG_basis, HG_basis, frames_per_sequence))

        for m in range(HG_basis):
            for n in range(HG_basis):
                # we saved oscilloscope files as sequence idx _ acquisition idx _ m _ n
                HD_non_amplified, HD_amplified, aom, trig_frame = \
                data_channels_from_binary(exp_folder+str(seq_n)+'_'+str(m)+'_'+str(n)+'.bin', scope_trace_points)

                start, stop = sequence_bounds_from_trig_frame(trig_frame)

                HD_non_amplified, HD_amplified, aom, trig_frame = HD_non_amplified[start:stop], HD_amplified[start:stop], \
                                                                    aom[start:stop], trig_frame[start:stop]    

                frames_start, frames_stop, n_frames = frames_start_stop(trig_frame) 

                j_arr_a, j_arr_na, ao_arr, j_chosen = [None] * n_frames, [None] * n_frames, [None] * n_frames, [None] * n_frames

                for i in range(frames_per_sequence):
                    #print('sequence: '+str(seq_n)+', mode: ('+str(m)+','+str(n)+')'+' '+str(i))
                    j_arr_a[i] = HD_amplified[frames_start[i]:frames_stop[i]]
                    j_arr_na[i] = HD_non_amplified[frames_start[i]:frames_stop[i]]
                    ao_arr[i] = aom[frames_start[i]:frames_stop[i]]                  

                    if np.size(np.where(j_arr_a[i] > 32256)) > 0: # if CH1 saturates, use CH0
                        j_chosen[i] = j_arr_na[i]
                    else:
                        j_chosen[i] = j_arr_a[i]*ch1_vert_scale/ch0_vert_scale
                       
                if parallel_processing:
                    # fit using parallel processing
                    phase_amp_seq_tuple = Parallel(
                        n_jobs=num_cores)(delayed(phase_and_amplitude_from_fit)(
                        j=j_chosen[i],
                        ao=ao_arr[i],
                        guess_fit_freq=guess_fit_freq,
                        debug=debug) for i in range(0, frames_per_sequence))

                    phase_tuple, amp_tuple = list(zip(*phase_amp_seq_tuple)) 
                    phase_diff[m,n,:], scope_amplitudes[m, n, :] = np.array(phase_tuple), np.array(amp_tuple)

                else:
                    for frame_idx in range(frames_per_sequence):
                        phase_diff[m,n,frame_idx], scope_amplitudes[m, n,frame_idx] = \
                            phase_and_amplitude_from_fit(j=j_chosen[frame_idx],
                                                         ao=ao_arr[frame_idx],
                                                         guess_fit_freq=guess_fit_freq,
                                                         debug=debug)


        phase_diff = np.nan_to_num(phase_diff)
        scope_amplitudes = np.nan_to_num(scope_amplitudes)

        if save_folder is not None:
            torch.save(torch.from_numpy(phase_diff), save_folder+'seq/'+str(seq_n)+'_phase.pt')
            torch.save(torch.from_numpy(scope_amplitudes), save_folder+'seq/'+str(seq_n)+'_amp.pt')

        phase_diff = phase_diff.reshape(HG_basis*HG_basis,frames_per_sequence)
        scope_amplitudes = scope_amplitudes.reshape(HG_basis*HG_basis,frames_per_sequence)

        
def phase_and_amplitude_from_fit(j, ao, guess_fit_freq = 0.52*0.5, debug = False):
    '''
    Given photocurrent and AOM sinusoidal data, return their phase difference and the photocurrent amplitude. 
    '''
    # Remove first 30 points, which corresonds to DMD transiting from one frame to the next
    j = j[30:]
    
    
    signal, norm_factor_sig = norm_data(j)
    ao_ref, _ = norm_data(ao) # 79.8 µs ± 9.88 µs
    
    signal_best_vals = fit_noisy_sin(signal, guess_fit_freq, debug=debug)
    ao_best_vals = fit_noisy_sin(ao_ref, guess_fit_freq, debug=debug)

    if (signal_best_vals is None) or (ao_best_vals is None):
        return 0,0 # we will look for exact '0' in the j tensors to see where fit fail..
    
    # phase difference between heterodyne and ao fits
    signal_freq_fit, signal_fitted_phase, signal_fitted_amplitude = \
    signal_best_vals[1], signal_best_vals[2], signal_best_vals[0]
    
    ao_freq_fit, ao_fitted_phase, ao_fitted_amplitude  = \
    ao_best_vals[1], ao_best_vals[2], ao_best_vals[0]
    
    signal_effective_phase = effective_phase(signal_fitted_phase, signal_fitted_amplitude)
    ao_effective_phase = effective_phase(ao_fitted_phase, ao_fitted_amplitude)
    j_phase_diff = signal_effective_phase - ao_effective_phase
    
    return j_phase_diff, np.abs(signal_fitted_amplitude*norm_factor_sig)
      
                                
def fit_noisy_sin(signal, guess_fit_freq, debug=False, error_treshold=0.5):
    '''
    Fit a noisy sinusoid and return fit parameters. 
    If fit fails or signal-fit error too high, return None.
    '''
    x = np.arange(0, signal.shape[0]) # 1.77 µs ± 433 ns
    
    try:
        signal_best_vals, signal_covar = curve_fit(cosinusoid_with_bias,
                                                   x,
                                                   signal,
                                                   p0 = [np.max(signal), guess_fit_freq, 0, 0]); # 1.45 ms ± 119 µs 
        signal_fit = cosinusoid_with_bias(x,
                                          signal_best_vals[0],
                                          signal_best_vals[1],
                                          signal_best_vals[2],
                                          signal_best_vals[3])# 18.3 µs ± 3.84 µs
    except (ValueError, RuntimeError): # fit fail
        return None

    signal_fit_std = np.sqrt(np.diag(signal_covar))[2] # check fit error

    if debug: # to implement, plot fit and data..
        signal_fit = cosinusoid_with_bias(x,
                                          signal_best_vals[0],
                                          signal_best_vals[1],
                                          signal_best_vals[2],
                                          signal_best_vals[3]) # 18.3 µs ± 3.84 µs
        
        print(signal_best_vals[1])
        plt.plot(signal, '.-')
        plt.plot(signal_fit, '-')
        plt.title('fit error: {:.2f}'.format(signal_fit_std))
        plt.show()

    if signal_fit_std > error_treshold:
        return None
    
    return signal_best_vals

                                
def effective_phase(phase, amplitude):
    ''' Get a sinusoid 'effective' phase, i.e. that keeps into account the amplitude sign, 
    from its phase and (positive or negative) amplitude. '''
    return phase + 0.5*np.pi*(-1+np.sign(amplitude))

##############################################################################################
                                
                                
if __name__ == '__main__': main()            