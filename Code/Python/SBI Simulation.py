"SBI Simulation"
#######################################################################################
# Module importation

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import numba
import time

# Working directory 
cwd = os.getcwd()
cwd

# Ploting parameters
plt.style.use('seaborn-talk');
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor']='#333F4B'
plt.rcParams['axes.linewidth']=0.8
plt.rcParams['xtick.color']='#333F4B'
plt.rcParams['ytick.color']='#333F4B'

from tqdm import tqdm 
import seaborn as sns
import pandas as pd

import torch
import sbi 

import sbi.inference
from sbi.inference.base import infer

from sbi.inference import SNPE, SNLE, SNRE, prepare_for_sbi ,simulate_for_sbi
from sbi.inference import likelihood_estimator_based_potential, DirectPosterior, MCMCPosterior, VIPosterior

from sbi.analysis import ActiveSubspace, pairplot
import sbi.utils as utils
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

# Suppression of the warning :
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action="ignore", category=FutureWarning)
np.seterr(all = 'ignore')

# Data-features
from DCM_region_features import calculate_summary_statistics

#######################################################################################
# Importation of the model : 
from DCM_region_model import Sigmodal
from DCM_region_model import DCM_Region #theta, constants, x_init, sig, eps, dt, ts, input_par
# DCM_Region = numba.jit(DCM_Region)

#######################################################################################
@jit(nopython=False)
def DCM_NMM_ERP_simulator_wrapper(params):
    
    params = np.asarray(params)

    # Time series parameters
    tend = 200.0
    dt=0.1
    t0=0.0
    ts = np.arange(t0, tend + dt, dt) # Time series array 
    nt=ts.shape[0]
    nn=int(9)
    nr=int(np.sqrt(int(params.shape[0])/3)) # Number of region
    x_init=np.zeros((nn)) # Initial condition of every equations

    delta=12.13
    alpha=-0.56 # Constante permet de definir la forme de la sigmoïde 
    # Parameter of the neuronal population
    g_1=0.42 
    g_2=0.76
    g_3=0.15
    g_4=0.16
    tau_e=5.77
    h_e=1.63
    h_i=27.87
    tau_i=7.77
    constants = np.array([g_1, g_2, g_3, g_4, delta, tau_i, h_i, tau_e, h_e, alpha])

    eps=0.
    sig=0.0
    #Stim parameter
    stim_init=np.round(ts.shape[0]*0.1) # in ms 
    stim_dur=40 # in ms
    u=3.94   
    input_par = np.array([stim_init,stim_dur,u])
    Sim_,stim_par_wrapper = DCM_Region(params, constants, x_init, sig, eps, dt, ts,input_par)

    Simulated_ERP=Sim_.reshape(nr,nn,nt)  
    Simulated_ERP_Pyramid=Simulated_ERP[:,8,:]
     
    return Simulated_ERP_Pyramid.reshape(-1)#,stim_par_wrapper

#######################################################################################
# Importation of the theta matrix:
# matrix size 3 nr*nr with nr the number of region
datapath=os.path.join(cwd+"/"+"matrix","Forward_3_region.npy")
matrice=np.load(datapath)
nr=int(matrice.shape[2])
theta_region = matrice.reshape(-1)

_=DCM_NMM_ERP_simulator_wrapper(theta_region)
features_=calculate_summary_statistics(_, features=['higher_moments',  'signal_power', 'signal_envelope', 'autocorlation', 'signal_peaks'])
features_.shape # Shape of 3 x 67, 67 features per regions

#######################################################################################
# Parameter for the many simulations
pmin=0
pmax=1.5
prior_min = np.ones((nr*nr*3))* pmin# Number of region
prior_max = np.ones((nr*nr*3))* pmax

num_params=len(prior_min)
prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))

# Number of simulations
num_sim=10

# Save path 
mysavepath= os.path.join(cwd+'/'+"Sim_data" , 'simulated_theta_x_dcm_region_allparam_connectivity_matrix_t200msec_'+str(num_sim)+'sims.npz')
#######################################################################################
# Simulation function :
def Save_simulations(simulator: Callable, prior,num_simulations: int):

    simulator, prior = prepare_for_sbi(simulator, prior)
    
    theta, x = simulate_for_sbi(simulator=simulator,
                                proposal=prior,
                                num_simulations=num_simulations,
                                show_progress_bar=True,)

    print( 'theta shape:',theta.shape,flush=True)
    print('data shape:', x.shape,flush=True)
    np.savez(mysavepath,theta=theta,x=x)

#######################################################################################
# Simulation init :
Save_simulations(DCM_NMM_ERP_simulator_wrapper, prior, num_simulations=num_sim)