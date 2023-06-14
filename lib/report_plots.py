#!/usr/bin/env python3
"""
@author: meysamhashemi INS Marseille

"""
        

import os
import sys
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import pickle

import re
import glob
import itertools
from itertools import chain
from operator import itemgetter
from pandas.plotting import scatter_matrix


#from lib.report_metrics import LSE, Err, RMSE, LSE_obs, z_score, shrinkage

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


from scipy.stats import gaussian_kde
from sbi.analysis.plot import _get_default_opts, _update, ensure_numpy



def get_limits(samples, limits=None):

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
    limits = np.asarray(limits)

    return limits
##########################################################################################################
def posterior_peaks(samples, return_dict=False, **kwargs):

    opts = _get_default_opts()
    opts = _update(opts, kwargs)

    limits = get_limits(samples)
    #samples = samples.numpy()
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
##########################################################################################################
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        my_ax.annotate("{:.2f}".format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=40)
##########################################################################################################
def plot_corr(x, y, **kwargs):
    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))
    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy = (0.2, 1.1), size = 20, xycoords = ax.transAxes)
##########################################################################################################
def plot_erp_fitted(ts, xpy_obs, xpy_est_mean):
    #plt.figure(figsize=(10, 6))
    plt.plot(ts, xpy_obs, lw=1, color='b', marker = '.', label='observed');
    plt.plot(ts, xpy_est_mean, lw=2, color='r' ,label='Fitted');
    plt.ylabel('Voltage[mV]', fontsize=22); 
    plt.xlabel('Time [ms]', fontsize=22); 
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16, frameon=False)
    plt.tight_layout()
##########################################################################################################
def plot_erp_ppc(ts, xpy_obs, x_ppc_lo, x_ppc_hi):
    #plt.figure(figsize=(10, 6))
    plt.plot(ts, xpy_obs, lw=1, color='b', marker = '.', label='observed');
    fill_between(ts, x_ppc_lo, x_ppc_hi, linewidth=2, alpha=0.7, facecolor='cyan', edgecolor='cyan', zorder=4, label='PPC')
    plt.ylabel('Voltage[mV]', fontsize=22); 
    plt.xlabel('Time [ms]', fontsize=22); 
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=16, frameon=False)
    plt.tight_layout()
##########################################################################################################
def plot_erp_fitted_ppc(ts, xpy_obs, x_ppc_lo, x_ppc_hi, xpy_mean, xpy_map, alg):
    #plt.figure(figsize=(10, 6))
    plt.plot(ts, xpy_obs, lw=1, color='b', marker = '.', zorder=1, label='observed');
    fill_between(ts, x_ppc_lo, x_ppc_hi, linewidth=2, alpha=0.7, facecolor='cyan', edgecolor='cyan', zorder=2, label='PPC')
    plt.plot(ts, xpy_mean, lw=2, color='r' , zorder=3, label=str(alg)+' (Mean)');
    plt.plot(ts, xpy_map, lw=2, color='gold' , zorder=4, label=str(alg)+' (MAP)');
    plt.ylabel('Voltage[mV]', fontsize=22); 
    plt.xlabel('Time [ms]', fontsize=22); 
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=15, frameon=False)
    plt.tight_layout()
##########################
def plot_true_est_map(theta_true, theta_est, theta_true_label):
    color = ['orange', 'y',  'olive', 'teal',  'red', 'lime', 'green', 'skyblue', 'royalblue',  'm']

    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(len(theta_true_label)):
            ax.scatter(theta_true[i], theta_est[i], s=30*(i+1), c=color[i],   lw=2,  label=theta_true_label[i], facecolors='none', edgecolors='none')
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()]),]
    ax.plot(lims, lims, linestyle= '--', color='grey', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.ylabel('$\Theta_{est}$', fontsize=22); 
    plt.xlabel('$\Theta_{true}$', fontsize=22); 
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=12, frameon=False)
    plt.axis('equal')
    plt.tight_layout()

##########################################################################################################
def plot_post_violin (theta_posterior, theta_true, theta_true_label):
    nn=int(theta_true.shape[0])
    #plt.figure(figsize=(10,4))
    plt.violinplot(theta_posterior, widths=0.7, showmeans=True, showextrema=True);
    plt.plot(np.arange(1,nn+1),theta_true ,'o', color='r', alpha=0.9, markersize=8);
    plt.xticks(np.arange(1,nn+1), theta_true_label, fontsize=20);
    plt.yticks(fontsize=18)
    plt.ylabel(' Posterior ' +r'${\Theta}$', fontsize=24);  
    plt.xlabel(r'${\Theta}$', fontsize=24); 
    plt.tight_layout()

##########################################################################################################
def plot_bar_estimations(y_est, y_err, theta_true, theta_true_label, alg):
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            my_ax.annotate("{:.2f}".format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=40)

    labels = theta_true_label
    error_kw= dict(ecolor='y', capsize=3, elinewidth=3)
    x = np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars

    # y_est=np.mean(np.mean(theta_est_hmc_convg, axis=0), axis=1)
    # y_err=np.std(np.mean(theta_est_hmc_convg, axis=0), axis=1)
    #y_est=np.mean(np.asarray([posterior_peaks(theta_est_hmc_convg[i,:,:].T, return_dict=False) for i in range(0,n_hmc_convg)]), axis=0)
    #y_err=np.std(np.asarray([posterior_peaks(theta_est_hmc_convg[i,:,:].T, return_dict=False) for i in range(0,n_hmc_convg)]), axis=0)

    fig, my_ax = plt.subplots(figsize=(18, 6))
    rects1 = my_ax.bar(x - width/2, theta_true, width, label='True')
    rects2 = my_ax.bar(x + width/2, y_est, yerr=y_err, width=0.3 ,alpha=0.8, error_kw=error_kw, label=str(alg))
    #plt.bar(range(len(y)), y_est, yerr=y_err, width=0.3 ,color='royalblue', alpha=0.8, error_kw=error_kw)
    my_ax.set_ylabel('Values', fontsize=20)
    my_ax.set_title(str(alg) + ' estimation', fontsize=20)
    my_ax.set_xticks(x)
    my_ax.set_xticklabels(labels,fontsize=20)
    my_ax.legend(fontsize=16, frameon=False)
    autolabel(rects1)
    autolabel(rects2)
    x0_, x1_, y0_, y1_ = plt.axis()
    plt.axis((x0_ - 0.,x1_ + 0.,y0_ - 0, y1_ + 5.))
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.tight_layout()








##########################################################################################################
def plot_post_dist(theta_est_chains, theta_true, theta_true_label):
    nr=int(len(theta_true_label)/3)
    for i in range(len(theta_true_label)):
            ax=plt.subplot(3,nr,i+1)
            ax=sns.kdeplot(x=theta_est_chains[:,i], color='blue',  lw=2)
            plt.axvline(x=theta_true[i], color='red', linestyle='--', lw=2)
            plt.axvline(x=posterior_peaks(theta_est_chains, return_dict=False)[i], color='cyan', linestyle='--', lw=1.5)
            if i==0 or i==nr or i==nr*2:
                plt.ylabel('Density', fontsize=18);  
            plt.xlabel((theta_true_label[i]), fontsize=18); 
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)            
    plt.tight_layout()
##########################################################################################################
def plot_post_dist_allchains(theta_est_convg, theta_true, theta_true_label, n_convg):
    color=[ 'lavender', 'lightblue', 'mediumslateblue' ,'royalblue', 'blue', 'b', 'darkblue', 'navy']

    #plt.figure(figsize=(12,4))
    grid = plt.GridSpec(2, 5)  
    for j in range(n_convg):
        i=0
        for name in theta_true_label:
                     plt.subplot(2,5,i+1)
                     ax=sns.kdeplot(x=theta_est_convg.T[:,i,j], color=color[j],  lw=2)
                     ax.set(ylabel=None)
                     plt.axvline(x=theta_true[i], color='red', linestyle='--', lw=2)
                     plt.axvline(x=posterior_peaks(theta_est_convg.T[:,:,j], return_dict=False)[i], color='cyan', linestyle='--', lw=1.5)
                     if i==0 or i==5:
                         plt.ylabel('Density', fontsize=18);  
                     plt.xlabel((theta_true_label[i]), fontsize=18); 
                     i=i+1
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)            
    plt.tight_layout()
##########################################################################################################
def plot_corr_estimations(theta_corr, theta_true_label):
    prop_coef=np.shape(theta_corr)[0]
    sns.set_style('white')
    mask = np.zeros_like(theta_corr)
    mask[np.triu_indices_from(mask)] = True
    fig = plt.figure(figsize=(10*prop_coef, 6*prop_coef))
    ax = fig.add_subplot(111)
    #cmap = sns.diverging_palette(240, 10, sep=20, as_cmap=True)
    sns.heatmap(abs(theta_corr), mask=mask, annot=True, robust=True, cmap='PuRd', linewidths=.0, annot_kws={'size':14}, fmt=".2f", vmin=0, vmax=1, ax=ax, xticklabels=theta_true_label, yticklabels=theta_true_label)
    #cmap='twilight_shifted'
    #ax.set_xticklabels()
    #ax.set_yticklabels()
    for label in ax.get_yticklabels():
         label.set_rotation(0)
    for label in ax.get_xticklabels():
         label.set_rotation(0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=50)
    ax.tick_params(labelsize=50)
    plt.tight_layout()

##########################################################################################################
def plot_jointpost_estimations (df_posterior, theta_true):
    nn=int(theta_true.shape[0])
    my_plot = sns.PairGrid(df_posterior)
    my_plot = my_plot.map_diag(plt.hist, bins = 10, edgecolor =  'k', color = 'blue', alpha=0.5, zorder=4);
    #my_plot = my_plot.map_diag(sns.kdeplot, shade=True)
    my_plot.palette = sns.color_palette("Set2", len(my_plot.hue_names))
    my_plot = my_plot.map_lower(plt.scatter, color = 'darkred', alpha=0.1)
    my_plot = my_plot.map_lower(plot_corr)
    my_plot = my_plot.map_upper(sns.kdeplot, cmap = 'Reds')
    #my_plot = my_plot.map_upper(sns.regplot, scatter_kws={'alpha':0.3})
    #my_plot.axes[0,0].set_ylim(0,.1);
    i=0
    for ax_ in my_plot.axes.ravel():
        if i in  [i*(nn+1) for i in range(0,nn,1)]:
           ax_.axvline(x=theta_true[int(i/10)], ymin = 0.0, ymax = 1, ls='--', linewidth=3, c='red', zorder=100) 
        i=i+1
    plt.tight_layout()

##########################################################################################################
def plot_jointpost_offdiag_estimations(df_posterior, theta_true):
    Axes=pd.plotting.scatter_matrix(df_posterior, alpha=0.5, linewidth=14, figsize=(12, 12), diagonal='kde')
    [plt.setp(item.yaxis.get_majorticklabels(), 'size', 10) for item in Axes.ravel()]
    [plt.setp(item.xaxis.get_majorticklabels(), 'size', 10) for item in Axes.ravel()]
    [plt.setp(item.yaxis.get_label(), 'size', 18) for item in Axes.ravel()]
    [plt.setp(item.xaxis.get_label(), 'size', 18) for item in Axes.ravel()];
    #plt.setp(ax.get_xticklabels(), rotation=90)
    for i in range(np.shape(Axes)[0]):
        for j in range(np.shape(Axes)[1]):
            if i < j:
                Axes[i,j].set_visible(False)
    i=0
    for ax__ in diag(Axes): 
             x_ = ax__.lines[0].get_xdata()
             y_ = ax__.lines[0].get_ydata()
             ymax = max(y_)
             xpos = np.argmax(y_, axis=0)
             xmax = x_[xpos]
             ax__.vlines(x=theta_true[i], ymin=0., ymax=y_.max(), colors='red', linestyle='--', label='Truth')
             ax__.vlines(x=xmax, ymin=0., ymax=y_.max(), colors='cyan', label='MAP') 
             i=i+1
    for ax__ in Axes.flatten():
        ax__.xaxis.label.set_rotation(0)
        ax__.xaxis.set_tick_params(rotation=90)
        ax__.yaxis.label.set_rotation(0)
        ax__.yaxis.label.set_ha('right')
    # for label in ax__.get_yticklabels():
    #      label.set_rotation(0)
    plt.tight_layout()
##########################################################################################################
def plot_elbo(Elbo):
    #plt.figure(figsize=(8, 4))
    plt.plot(Elbo)
    plt.ylabel('ELBO', fontsize=22); 
    plt.xlabel('100*iter', fontsize=22); 
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
##########################################################################################################
def plot_degeneracy_timesacels(theta_posterior, theta_true_label):
    import matplotlib.gridspec as gridspec
    from lib.report_plots import SeabornFig2Grid

    fig = plt.figure(figsize=(14,5))
    gs = gridspec.GridSpec(1, 3)

    g0 = sns.jointplot(theta_posterior[:,5], theta_posterior[:,4], kind='scatter',color='purple',  size=8, space=0,ratio=3,) 
    g0.set_axis_labels(theta_true_label[5], theta_true_label[4], fontsize=20)
    g0.plot_joint(sns.kdeplot, color="r", zorder=4, levels=1)


    g1 = sns.jointplot(theta_posterior[:,7], theta_posterior[:,4], kind='scatter',color='purple',  size=8, space=0,ratio=3,) 
    g1.set_axis_labels(theta_true_label[7], theta_true_label[4], fontsize=20)
    g1.plot_joint(sns.kdeplot, color="r", zorder=4, levels=1)

    g2 = sns.jointplot(theta_posterior[:,7], theta_posterior[:,5], kind='scatter',color='purple',  size=8, space=0,ratio=3,) 
    g2.set_axis_labels(theta_true_label[7], theta_true_label[5], fontsize=20)
    g2.plot_joint(sns.kdeplot, color="r", zorder=4, levels=1)



    mg0 = SeabornFig2Grid(g0, fig, gs[0])
    mg1 = SeabornFig2Grid(g1, fig, gs[1])
    mg2 = SeabornFig2Grid(g2, fig, gs[2])

    gs.tight_layout(fig)
##########################################################################################################
def plot_degeneracy_synapses(theta_posterior, theta_true_label):

    fig = plt.figure(figsize=(14,4))
    gs = gridspec.GridSpec(1, 4)

    g0 = sns.jointplot(theta_posterior[:,8], theta_posterior[:,0], kind='scatter',color='purple',  size=6, space=0,ratio=3,) 
    g0.set_axis_labels(theta_true_label[8], theta_true_label[0], fontsize=20)
    g0.plot_joint(sns.kdeplot, color="r", zorder=4, levels=1)


    g1 = sns.jointplot(theta_posterior[:,8], theta_posterior[:,1], kind='scatter',color='purple',  size=6, space=0,ratio=3,) 
    g1.set_axis_labels(theta_true_label[8], theta_true_label[1], fontsize=20)
    g1.plot_joint(sns.kdeplot, color="r", zorder=4, levels=1)


    g2 = sns.jointplot(theta_posterior[:,8], theta_posterior[:,2], kind='scatter',color='purple',  size=6, space=0,ratio=3,) 
    g2.set_axis_labels(theta_true_label[8], theta_true_label[2], fontsize=20)
    g2.plot_joint(sns.kdeplot, color="r", zorder=4, levels=1)

    g3 = sns.jointplot(theta_posterior[:,6], theta_posterior[:,3], kind='scatter',color='purple',  size=6, space=0,ratio=3,) 
    g3.set_axis_labels(theta_true_label[6], theta_true_label[3], fontsize=20)
    g3.plot_joint(sns.kdeplot, color="r", zorder=4, levels=1)


    mg0 = SeabornFig2Grid(g0, fig, gs[0])
    mg1 = SeabornFig2Grid(g1, fig, gs[1])
    mg2 = SeabornFig2Grid(g2, fig, gs[2])
    mg3 = SeabornFig2Grid(g3, fig, gs[3])

    gs.tight_layout(fig)
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
  
##########################################################################################################      

##########################################################################################################
def plot_zscore_shrinkage(theta_true_mu, theta_est_mu, theta_est_std, prior_std, theta_true_label):
    z_score_theta=z_score(theta_true_mu, theta_est_mu, theta_est_std)
    colors= ['orange', 'y', 'teal',  'red', 'lime', 'green', 'skyblue', 'royalblue',  'm']*3
    marker=["o"]*9+["h"]*9+["*"]*9
    fig,ax = plt.subplots()
    for i in range(len(theta_true_label)):
        ax.scatter(shrinkage(prior_std[i], theta_est_std[i]), z_score_theta[i] ,s=300, marker=marker[i], c=colors[i])
    fig.legend(labels=theta_true_label, fontsize=12, frameon=False)    
    ax.set_xlabel("Posterior shrinkages", fontsize=22)
    ax.set_ylabel("Posterior z-scores", fontsize=22)
    ax.tick_params(labelsize=20)
    #ax.set_yticks(fontsize=18)
    #plt.axis((0,1.1,0,10))
##########################################################################################################
# def plot_zscore_shrinkage(nodes, eta_true_mu, eta_est_mu, eta_est_std, prior_std):
#     z_score_eta=z_score(eta_true_mu, eta_est_mu, eta_est_std)
#     colors= np.random.rand(z_score_eta.shape[0])
#     plt.scatter(shrinkage([prior_std]*nodes.shape[0], eta_est_std), z_score_eta ,s=120, c='blue')
#     plt.xlabel("Posterior shrinkages", fontsize=14)
#     plt.ylabel("Posterior z-scores", fontsize=14)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.axis((0,1.1,0,10))
    #plt.text(-.4, 10.4, "C" ,fontsize=24, fontweight='bold')
##########################################################################################################
# def pair_plots_params(csv, keys, skip=0):
#     n = len(keys)
#     if isinstance(csv, dict):
#         csv = [csv]  # following assumes list of chains' results
#     for i, key_i in enumerate(keys):
#         for j, key_j in enumerate(keys):
#             plt.subplot(n, n, i*n+j+1)
#             for csvi in csv:
#                 if i==j:
#                     plt.hist(csvi[key_i][skip:], 20, log=False)
#                     plt.xticks(fontsize = 10)   
#                     plt.yticks(fontsize = 10)  
#                 else:
#                     plt.plot(csvi[key_j][skip:], csvi[key_i][skip:], '.')
#                     plt.xticks(fontsize = 10)   
#                     plt.yticks(fontsize = 10)  
#             if i==0:
#                 plt.title(key_j, fontsize = 14)
#             if j==0:
#                 plt.ylabel(key_i, fontsize = 14)

def pair_plots_params(csv, keys, skip=0):
    n = len(keys)
    if isinstance(csv, dict):
        csv = [csv]  # following assumes list of chains' results
    for i, key_i in enumerate(keys):
        for j, key_j in enumerate(keys):
            plt.subplot(n, n, i*n+j+1)
            for csvi in csv:
                if i==j:
                    plt.hist(csvi[key_i][skip:], 20, log=False)
                    plt.xticks([])   
                    plt.yticks([]) 
                    plt.axis('off')
                else:
                    plt.plot(csvi[key_j][skip:], csvi[key_i][skip:], '.')
                    plt.xticks([])   
                    plt.yticks([])  

            if i==0:
                plt.title(key_j, fontsize = 14)
                plt.xticks([])   
                plt.yticks([])  

            if j==0:
                plt.ylabel(key_i, fontsize = 14)
                plt.xticks([])   
                plt.yticks([])  
##########################################################################################################
def pair_plots(samples, params, figname='', sampler=None):
    import numpy as np
    import matplotlib.pyplot as plt
    div_iters = np.where(samples['divergent__'] == 1)[0] if sampler == 'HMC' else []
    plt.figure(figsize=(23, 13))
    nParams = len(params)
    for i in range(nParams):
        for j in range(nParams):
            plt.subplot(nParams, nParams, i * nParams + (j + 1))
            if (i == j):
                plt.hist(samples[params[i]].flatten(), bins=20, color='black')
            else:
                xvals = np.mean(
                    samples[params[j]], axis=1) if len(
                        samples[params[j]].shape) > 1 else samples[params[j]]
                yvals = np.mean(
                    samples[params[i]], axis=1) if len(
                        samples[params[i]].shape) > 1 else samples[params[i]]
                for k in range(xvals.shape[0]):
                    if (k in div_iters):
                        plt.plot(xvals[k], yvals[k], 'ro', alpha=0.8)
                    else:
                        plt.plot(xvals[k], yvals[k], 'ko', alpha=0.1)
            if (i == 0):
                plt.title(params[j], fontsize=13)
            if (j == 0):
                plt.ylabel(params[i], fontsize=13, rotation=90)
    plt.tight_layout()
    if (figname):
        plt.savefig(figname)                
##########################################################################################################
##########################################################################################################
##########################################################################################################        
#From scikit-learn
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """


    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    #plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.title('Confusion matrix', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('True nodes', fontsize=12)
    plt.xlabel('Predicted nodes\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), fontsize=12)
    #plt.text(-1.8, -.6, "D" ,fontsize=24, fontweight='bold')
##########################################################################################################        
##########################################################################################################        

import matplotlib.gridspec as gridspec


class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

##########################################################################################################        
########################################################################################################## 