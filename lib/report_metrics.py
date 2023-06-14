#!/usr/bin/env python3
"""
@author: meysamhashemi INS Marseille

"""
import os
import sys
import numpy as np
import re
#####################################################
def LSE(x1, x2):
    return np.sum((x1 - x2)**2)
#####################################################
def Err(x1, x2):
    return np.sum(np.abs(x1 - x2))
#####################################################    
def RMSE(x1, x2):
    return np.sqrt(((x1 - x2) ** 2).mean()) 
#####################################################
def LSE_obs(Obs, Obs_lo, Obs_hi):
    return np.average([LSE(Obs, Obs_lo), LSE(Obs, Obs_hi)])
#####################################################
def z_score(true_mean, post_mean, post_std):
    return np.abs((post_mean - true_mean) / post_std)
#####################################################
def shrinkage(prior_std, post_std):
    return 1 - (post_std / prior_std)**2
#####################################################
import torch
import numpy as np
from scipy.stats import gaussian_kde
from sbi.analysis.plot import _get_default_opts, _update, ensure_numpy


def _get_limits(samples, limits=None):

    if type(samples) != list:
        samples = ensure_numpy(samples)
        samples = [samples]
    else:
        for i, sample_pack in enumerate(samples):
            samples[i] = ensure_numpy(samples[i])

    # Dimensionality of the problem.
    dim = samples[0].shape[1]

    if limits == [] or limits is None:
        limits = []
        for d in range(dim):
            min = +np.inf
            max = -np.inf
            for sample in samples:
                min_ = sample[:, d].min()
                min = min_ if min_ < min else min
                max_ = sample[:, d].max()
                max = max_ if max_ > max else max
            limits.append([min, max])
    else:
        if len(limits) == 1:
            limits = [limits[0] for _ in range(dim)]
        else:
            limits = limits
    limits = torch.as_tensor(limits)

    return limits


def posterior_peaks(samples, return_dict=False, **kwargs):
    '''
    Finds the peaks of the posterior distribution.

    Args:
        samples: torch.tensor, samples from posterior
    Returns: torch.tensor, peaks of the posterior distribution
            if labels provided as a list of strings, and return_dict is True
            returns a dictionary of peaks

    '''

    opts = _get_default_opts()
    opts = _update(opts, kwargs)

    limits = _get_limits(samples)
    samples = np.asarray(samples)
    n, dim = samples.shape

    try:
        labels = opts['labels']
    except:
        labels = range(dim)

    peaks = {}
    if labels is None:
        labels = range(dim)
    for i in range(dim):
        peaks[labels[i]] = 0

    for row in range(dim):
        density = gaussian_kde(
            samples[:, row],
            bw_method=opts["kde_diag"]["bw_method"])
        xs = np.linspace(
            limits[row, 0], limits[row, 1],
            opts["kde_diag"]["bins"])
        ys = density(xs)

        # y, x = np.histogram(samples[:, row], bins=bins)
        peaks[labels[row]] = xs[ys.argmax()]

    if return_dict:
        return peaks
    else:
        return list(peaks.values())
