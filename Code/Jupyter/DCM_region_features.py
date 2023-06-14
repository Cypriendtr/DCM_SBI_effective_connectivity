#!/usr/bin/env python3
"""
@author: meysamhashemi  INS Marseille

"""
import os
import sys
import numpy as np
import scipy as scp

from scipy import signal
from scipy.signal import  find_peaks, peak_widths, savgol_filter
from scipy import stats as spstats
from scipy.stats import moment
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import mode
from scipy.signal import hilbert

from scipy.optimize import curve_fit

from sklearn.preprocessing import minmax_scale 

nr=int(3)

def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b



def envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 1).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < -1).nonzero()[0] + 1 
    

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]


    # global max of dmax-chunks of locals max 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global min of dmin-chunks of locals min 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
#     ts=np.array(0.1*np.r_[0:s.shape[0]])
#     plt.plot(ts,s,label='signal', color= 'b')
#     plt.plot(ts[high_idx], s[high_idx], 'g', label='low')
#     plt.plot(ts[low_idx], s[low_idx], 'r', label='high')

    return lmin,lmax




def calculate_summary_statistics(x_input, features):
        """Calculate summary statistics

        Parameters
        ----------
        x : output of the simulator

        Returns
        -------
        np.array, summary statistics
        """

        n_summary = 100
        
        x_input=x_input.reshape(3,int(x_input.shape[0]/3))

        #x= savgol_filter(x_input, 51, 1)    
        #x_n= minmax_scale(x, axis=1)
        

        full_stats_vec=np.array([])
        for i in range(nr):
                x=x_input[i]
                x_l=x[0:int(x.shape[0]/2)]
                x_r=x[int(x.shape[0]/2):int(x.shape[0])]
                sum_stats_vec = np.concatenate((np.array([np.mean(x)]),
                                                np.array([np.std(x)]),
                                                np.array([skew(x)]),
                                                np.array([kurtosis(x)]),
                                                np.array([np.max(x)]),
                                                np.array([np.min(x)]),
                                                np.array([np.argmin(x)]),
                                                np.array([np.argmax(x)]),
                                                np.array([np.mean(x_l)]),
                                                np.array([np.mean(x_r)]),
                                                np.array([np.var(x_l)]),
                                                np.array([np.var(x_r)]),
                                                np.array([np.max(x_l)]),
                                                np.array([np.max(x_r)]), 
                                                np.array([np.min(x_l)]),
                                                np.array([np.min(x_r)]),
                                                np.array([np.argmax(x_l)]),
                                                np.array([np.argmax(x_r)]),
                                                ))

                
                for item in features:

                        if item is 'higher_moments':

                                sum_stats_vec = np.concatenate((sum_stats_vec,
                                                                np.array([moment(x, moment=2)]),
                                                                np.array([moment(x, moment=3)]),
                                                                np.array([moment(x, moment=4)]),
                                                                np.array([moment(x, moment=5)]),
                                                                np.array([moment(x, moment=6)]),
                                                                np.array([moment(x, moment=7)]),
                                                                np.array([moment(x, moment=8)]),
                                                                np.array([moment(x, moment=9)]),
                                                                np.array([moment(x, moment=10)]),
                                                                                ))

                        if item is 'signal_power':
                                        
                                x_area = np.trapz(x, dx=0.1)
                                x_pwr = np.sum((x*x))
                                sum_abs=np.sqrt(np.sum(abs(x)))
                                mean_abs=np.mean(abs(x))

                                x_size = 2 ** np.ceil(np.log2(2*len(x) - 1)).astype('int')
                                x_var = np.var(x)
                                ndata = x - np.mean(x)
                                x_fft = np.fft.fft(ndata, x_size)
                                x_pwr_fft = np.sqrt(np.sum(np.abs(x_fft) ** 2))

                                fs = 10e3
                                f, Pxx_den =  signal.periodogram(x, fs)
                                Pxx_power= np.sqrt(np.sum(np.abs(Pxx_den) ** 2))

                                sum_stats_vec = np.concatenate((sum_stats_vec,
                                                                np.array([x_area]),
                                                                np.array([x_pwr]),
                                                                np.array([x_pwr_fft]),
                                                                np.array([Pxx_power]),
                                                                np.array([np.max(Pxx_den)]),
                                                                sum_abs.reshape(-1),
                                                                mean_abs.reshape(-1),
                                                                ))
                
                
                        if item is 'autocorlation':
                
                                x_mean = np.mean(x)
                                x_var = np.var(x)
                                ndata = x - x_mean
                                acorr = np.correlate(ndata, ndata, 'full')[len(ndata)-1:] 
                                x_acorr = acorr / x_var / len(ndata)
                                
                                x_acorr_l=x[0:int(x_acorr.shape[0]/2)]
                                x_acorr_r=x[int(x_acorr.shape[0]/2):int(x_acorr.shape[0])]
                
                                x_acorr_sum_abs=np.sqrt(np.sum(abs(x_acorr)))
                                x_acorr_mean_abs=np.mean(abs(x_acorr))
                                
                                x_selfcorr=np.mean(np.correlate(x, x, mode='same'))

                                x_acorr_peaks_max_, _ = find_peaks(x_acorr,  rel_height=.5, width=10, distance=200, prominence=.05)
                                x_acorr_peaks_min_, _ = find_peaks(-x_acorr, rel_height=.5, width=10, distance=200, prominence=.05)

                                if x_acorr_peaks_max_.shape[0] >= 2:
                                        x_acorr_peaks_max=x_acorr_peaks_max_[0:2]
                                else:
                                        x_acorr_peaks_max=np.zeros((2))
                
                                if x_acorr_peaks_min_.shape[0] >= 2:
                                        x_acorr_peaks_min=x_acorr_peaks_min_[0:2]
                                else:
                                        x_acorr_peaks_min=np.zeros((2))
                
                                                
                                sum_stats_vec = np.concatenate((sum_stats_vec,
                                                                np.array([np.std(x_acorr)]),
                                                                np.array([skew(x_acorr)]),
                                                                np.array([kurtosis(x_acorr)]),
                                                                np.array([np.mean(x_acorr_r)]),
                                                                np.array([np.var(x_acorr_r)]),
                                                                x_acorr_sum_abs.reshape(-1),
                                                                x_acorr_mean_abs.reshape(-1),
                                                                np.array([x_selfcorr]), 
                                                                np.array(x_acorr_peaks_max.reshape(-1)),
                                                                np.array(x_acorr_peaks_min.reshape(-1)),
                                                                np.array(x_acorr_peaks_max.reshape(-1)),

                                                                ))
                        
                                
                        if item is 'signal_envelope':
                        
                                low_idx, high_idx  = envelopes_idx(x)

                                if low_idx.shape[0]<=1:
                                        low_idx=np.append(low_idx, [int(np.zeros((1))), int(np.zeros((1)))])
                                if high_idx.shape[0]<=1:
                                        high_idx=np.append(high_idx, [int(np.zeros((1))), int(np.zeros((1)))])

                                sign_x = np.sign(x)
                                sign_x_flip=np.where(np.diff(np.sign(x)))[0]
                                index_pos = np.where(sign_x == +1)
                                index_neg = np.where(sign_x == -1)
                                                        
                                x_area_pos = np.trapz(x[index_pos])
                                x_area_neg = np.trapz(x[index_neg])
                                
                                if sign_x_flip.shape[0]<1:
                                        sign_x_flip=np.append(sign_x_flip, [int(np.zeros((1)))])
                                        
                                sum_stats_vec = np.concatenate((sum_stats_vec,
                                                                np.array([np.diff(low_idx)[0]]),
                                                                np.array([np.diff(high_idx)[0]]),
                                                                np.array([sign_x_flip.shape[0]]),
                                                                np.array([index_pos[0].shape[0]]),
                                                                np.array([index_neg[0].shape[0]]),
                                                                np.array([x_area_pos]),
                                                                np.array([x_area_neg]),
                                                                                ))

                        if item is 'signal_peaks':
                                        
                                        peak_to_peak=abs(np.max(x)-np.min(x))

                                        nt=x.shape[0]
                                        v=np.zeros(nt)
                                        v= np.array(x)

                                        v_th=0
                                        ind = np.where(v < v_th)
                                        v[ind] = v_th

                                        ind = np.where(np.diff(v) < 0)
                                        v[ind] = v_th

                                        peak_times = np.array(0.1*np.r_[0:nt])[ind]
                                        peak_times_times_stim = peak_times

                                        if peak_times_times_stim.shape[0] > 0:
                                                peak_times_times_stim = peak_times_times_stim[np.append(1, np.diff(peak_times_times_stim)) > 0.5]

                                        peaks_max_, _ = find_peaks(x,  rel_height=0.5, width=10, distance=200, prominence=.05)
                                        peaks_min_, _ = find_peaks(-x, rel_height=0.5, width=10, distance=200, prominence=.05)

                                        if peaks_max_.shape[0] >= 2:
                                                peaks_max=peaks_max_[0:2]
                                        else:
                                                peaks_max=np.zeros((2))
                
                                        if peaks_min_.shape[0] >= 2:
                                                peaks_min=peaks_min_[0:2]
                                        else:
                                                peaks_min=np.zeros((2))
                
                                        sum_stats_vec = np.concatenate((sum_stats_vec, 
                                                                        peak_to_peak.reshape(-1),
                                                                        np.array([peak_times_times_stim.shape[0]]),
                                                                        np.array([peaks_max_.shape[0]]),
                                                                        np.array([peaks_min_.shape[0]]),
                                                                        np.array(peaks_max.reshape(-1)),
                                                                        np.array(peaks_min.reshape(-1)),
                                                                        x[peaks_max.astype(int)],
                                                                        x[peaks_min.astype(int)],
                                                                ))
                                        
                sum_stats_vec = sum_stats_vec[0:n_summary]
                stats_shape=int(sum_stats_vec.shape[0])
                full_stats_vec=np.concatenate((full_stats_vec,sum_stats_vec))
                
        return full_stats_vec