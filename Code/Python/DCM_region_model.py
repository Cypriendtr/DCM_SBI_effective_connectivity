import numpy as np

#######################################################################################
#The Sigmodal function allow the transition from a membrane potential to a firing rate
def Sigmodal(x1, x2, delta, alpha):
    S=(1./(1.+ np.exp(alpha*(x1-(delta*x2)))))-0.5
    return S
#######################################################################################
# Function ODE for every region 
def DCM_Region(theta  ,constants , x_init, sig: float , eps:float, dt:float, ts, I, default=True):

    
    '''
	ARGUMENTS
	theta		: parameters to be optimized, Connectivity matrix shape (3 {type of projections} x nbr_region x nbr_region)
	constants 	: fixed parameters (not optimized)
	x_init 		: initial condition for state variable x
	sig		    : neuronal noise standard deviation
	eps		    : observation noise standard deviation
	dt 		    : time step for simulation
	ts		    : time stamps for simulation (number of time steps)
    I           : Input stimulation parameter, shape (t_init, t_end, u)
    default     : Define if the user want all the region to have the exact same constants parameters, by default True

	VARIABLES
	x correpsond to the state equations, the pyramidal neuron activity is x[i,8,:], which is the observed output (including noise related to eps)	
	'''

    # Parameter shapping 
    matrix=theta
    nr=int(np.sqrt(int(matrix.shape[0])/3)) # Number of regions
    matrix=matrix.reshape(3,nr,nr)
    nn=9 # Number of equations, always = 9
    nt=ts.shape[0] # Number of time step

    # Input parameter 
    stim_init=int(I[0]) # in ms 
    stim_end=int(stim_init+(I[1]/dt)) # in ms
    u=I[2] # Model the afferent connexion 

    # Stimulation init
    U=np.zeros((nr,nt))
    U[0,stim_init:stim_end]=u
    stim_par=np.hstack((np.array([stim_init,stim_end,u]),U[0])) # All the stimulation parameter

    # Integration parameter
    dt = np.float64(dt) # Time step
    sig = np.float64(sig) # 
    x = np.zeros((nr,nn, nt)) # Every equation time series
    xs = np.zeros((nr,nn, nt)) # Output vector 
    dx = np.zeros((nr,nn)) # Integration step
    x[0,:,0] = x_init 
    sys_noise = np.random.randn(nr,nn, nt) # Noise of the regiosn itself
    obs_noise = np.random.randn(nt) # Observator noise 

    # Formatting variables
    if default == True:
        g_1, g_2, g_3, g_4, delta, tau_i, h_i, tau_e, h_e, alpha,  = constants # All the columns have the same variables
        g_1=float(g_1)  
        g_2=float(g_2)
        g_3=float(g_3)
        g_4=float(g_4)
        delta=float(delta)
        tau_i=float(tau_i)
        h_i=float(h_i)
        tau_e=float(tau_e)
        h_e=float(h_e)
        alpha=float(alpha)
        # Integaration system 
        for t in range(0, nt-1):
                for i in range (nr):
                        dx[i,0] = x[i,3,t]
                        dx[i,1] = x[i,4,t]
                        dx[i,2] = x[i,5,t]
                        dx[i,3] = (1./tau_e) * (h_e * (np.sum((matrix[0,i] + matrix[2,i]) * Sigmodal(x[i,8,t],-x[:,8,t],delta,alpha)) + (g_1 * (Sigmodal(x[i,8,t], x[i,4,t] - x[i,5,t] , delta, alpha)) +U[i,t])) - (x[i,0,t]/tau_e) - 2 * x[i,3,t])
                        dx[i,4] = (1./tau_e) * (h_e * (np.sum((matrix[1,i] + matrix[2,i]) * Sigmodal(x[i,0,t],-x[:,8,t],delta,alpha)) + (g_2 * (Sigmodal(x[i,0,t], x[i,3,t], delta, alpha)))) - (x[i,1,t]/tau_e)-2 * x[i,4,t])
                        dx[i,5] = (1./tau_i) * (h_i * (g_4 * (Sigmodal(x[i,6,t], x[i,7,t], delta, alpha))) - (x[i,2,t]/tau_i)-2 * x[i,5,t])
                        dx[i,6] = x[i,7,t]
                        dx[i,7] = (1./tau_e) * (h_e * (np.sum((matrix[1,i] + matrix[2,i]) * Sigmodal(x[i,8,t],-x[:,8,t],delta,alpha)) + (g_3 * (Sigmodal(x[i,8,t], x[i,4,t] - x[i,5,t], delta, alpha)))) - (x[i,6,t]/tau_e) - 2 * x[i,7,t])
                        dx[i,8] = x[i,4,t] - x[i,5,t] 
                        x[i,:,t+1] = x[i,:,t] + dt * dx[i,:] + np.sqrt(dt) * sig * sys_noise[i,:,t]
        xs=x
        xs[:,8,:] = x[:,8,:] + eps * obs_noise # Inputing the observator noise

    elif default == False:
        # The constante variable must be shape 10 x nr for every constant variables
        def verif_constants(constants):
            assert constants.shape[1]==nr, "Constants shape must be : 10 x num_region" 
        try:
            verif_constants(constants)
        except:
            raise Exception( "Error in constants size")
        
        g_1=constants[0]
        g_2=constants[1]
        g_3=constants[2]
        g_4=constants[3]
        delta=constants[4]
        tau_i=constants[5]
        h_i=constants[6]
        tau_e=constants[7]
        h_e=constants[8]
        alpha=constants[9]
        # Integaration system 
        for t in range(0, nt-1):
                for i in range (nr):
                        dx[i,0] = x[i,3,t]
                        dx[i,1] = x[i,4,t]
                        dx[i,2] = x[i,5,t]
                        dx[i,3] = (1./tau_e[i]) * (h_e[i] * (np.sum((matrix[0,i] + matrix[2,i]) * Sigmodal(x[i,8,t],-x[:,8,t],delta[i],alpha[i])) + (g_1[i] * (Sigmodal(x[i,8,t], x[i,4,t] - x[i,5,t] , delta[i], alpha[i])) +U[i,t])) - (x[i,0,t]/tau_e[i]) - 2 * x[i,3,t])
                        dx[i,4] = (1./tau_e[i]) * (h_e[i] * (np.sum((matrix[1,i] + matrix[2,i]) * Sigmodal(x[i,0,t],-x[:,8,t],delta[i],alpha[i])) + (g_2[i] * (Sigmodal(x[i,0,t], x[i,3,t], delta[i], alpha[i])))) - (x[i,1,t]/tau_e[i])-2 * x[i,4,t])
                        dx[i,5] = (1./tau_i[i]) * (h_i[i] * (g_4[i] * (Sigmodal(x[i,6,t], x[i,7,t], delta[i], alpha[i]))) - (x[i,2,t]/tau_i[i])-2 * x[i,5,t])
                        dx[i,6] = x[i,7,t]
                        dx[i,7] = (1./tau_e[i]) * (h_e[i] * (np.sum((matrix[1,i] + matrix[2,i]) * Sigmodal(x[i,8,t],-x[:,8,t],delta[i],alpha[i])) + (g_3[i] * (Sigmodal(x[i,8,t], x[i,4,t] - x[i,5,t], delta[i], alpha[i])))) - (x[i,6,t]/tau_e[i]) - 2 * x[i,7,t])
                        dx[i,8] = x[i,4,t] - x[i,5,t] 
                        x[i,:,t+1] = x[i,:,t] + dt * dx[i,:] + np.sqrt(dt) * sig * sys_noise[i,:,t]
        xs=x
        xs[:,8,:] = x[:,8,:] + eps * obs_noise # Inputing the observator noise
        
    return xs.reshape(-1),stim_par