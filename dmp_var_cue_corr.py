import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from matplotlib import colors



def init_params(N=51, T=10, dt=0.1, init_state=[], rate01=2.0, rate10=1.0, rI=0.5, rC=0.5, 
                cue_updates=2, cue_correlation=1): 
    params={}
    params['N'] = N # number of individuals
    params['T'] = T # total integration time
    params['dt'] = dt # step size
    params["time_steps"] = int(T/dt)
    params['rate01'] = rate01 # rate of switching from independent to correlated cue
    params['rate10'] = rate10 # rate of switching from correlated to independent cue
    params['rI'] = rI # reliability of independent cue 
    params['rC'] = rC # reliability of correlated cue 
    params['cue_updates'] = cue_updates # averageupdate intervalls of correlated cue 
    params['cue_correlation'] = max((1/N), abs(cue_correlation)) # percent of individuals observing the same correalted cue ([0,1])
    
    if (init_state == []) or (len(init_state) != N): 
        init_state = np.ones(N)
        p_0=rate10/(rate01+rate10)
        rand = np.random.rand(N)
        init_state[rand<p_0] = -1
        
    params['init_state'] = init_state
    params['n_corr_cues'] = int(1/params['cue_correlation'])
    a = np.arange(N)
    np.random.shuffle(a)
    params['cue_assignments'] = [list(arr) for arr in np.array_split(a, params['n_corr_cues'])]

    
    return params 


def SingleStep(past_state,rate01,rate10,dt):
    ''' Calculates the state update for N indviduals from past_state to new_state'''
    
    N=len(past_state) # number of individuals 
    random_number=np.random.random(N)
    new_state=np.zeros(N)
    
    for n in range(N):
        # depending on the state set the right transition rate
        if(past_state[n]==-1):
            rate=rate01 # switch from correlated to independent
        else:
            rate=rate10 # swotch from indepenent to correlated 
            
        new_state[n]=past_state[n]    
        # check whether transition occurs and if so swap sign of the state    
        if(random_number[n]<rate*dt):
                new_state[n]*=-1
       
    return new_state

def collectiveDMP(initial_state,rate01,rate10,time_steps,dt=0.1):
    ''' Simple Run Generating N dimensional / N agents DMP '''
    N=len(initial_state)# number of individuals
    state=np.zeros((int(time_steps),N))
    #print(np.shape(state))
    state[0,:]=initial_state
    for s in range(1,int(time_steps)):
        state[s,:]=SingleStep(state[s-1],rate01,rate10,dt)
        
    return state

def get_ind_cue_values(params):
    ''' return values of the independent cue for N individuals over all time steps '''
    N = params["N"]
    time_steps = params["time_steps"]
    rI = params["rI"]
    
    return 2*(np.random.rand(time_steps, N)<rI).astype(int)-1


def get_corr_cue_values(params):
    ''' return values of the correlated cue which is updated at expoentially distributed intervalls (mean = cueUpdates)'''
    time_steps = params["time_steps"]
    rC = params["rC"]
    cue_updates = params["cue_updates"]
    Ncue = params['n_corr_cues']
    
    if cue_updates > time_steps: # in this cases the correlated cue remains constant
        return np.multiply(2*(np.random.uniform(size=Ncue)<rC).astype(int)-1, np.ones((time_steps, Ncue)))
    
    VALs = []
    for cue_idx in range(Ncue): 
        # draw waiting times
        n_updates = int(np.ceil(time_steps/cue_updates))
        wait_times = np.ceil(np.random.exponential(scale=cue_updates, size = n_updates)).astype(int)

        # calculate timepoints of updates  
        update_times = [0]
        for k in list(np.cumsum(wait_times)): 
            update_times.append(k)
        update_times.append(time_steps)

        # darw cue values
        values = np.zeros(time_steps)
        for i in range(len(update_times)-1): 
            values[update_times[i]:update_times[i+1]] = 2*int(np.random.uniform()<rC)-1
        
        VALs.append(values)
                        
    return np.array(VALs).reshape((time_steps, Ncue))


def RandomWalkDMP(params, return_everything=False):
    N = params['N']
    T = params['T']
    dt = params['dt']
    
    # draw time line of individuals' states
    states = collectiveDMP(params['init_state'],params['rate01'],params['rate10'],int(T/dt),dt)
    #print(states.shape)
    # draw time line of values of the independent cue 
    ind_vals = get_ind_cue_values(params)
    #print(ind_vals.shape)
    # draw values of the correlated cue
    corr_vals= get_corr_cue_values(params)
    #print(corr_vals.shape)
    
    
    # rescale state values {-1, 1} <- {0,1}
    states= 0.5*(1+states)# 1 is independent 0 is correlated

    # claculate the individuals' positions based on the independent cue 
    positions_ind = np.multiply(states, ind_vals)*dt
    
    # claculate the individuals' positions based on the currelated cue 
    positions_corr = np.zeros((params["time_steps"], params["N"]))
    for n_cue in range(params['n_corr_cues']): 
        idx = params["cue_assignments"][n_cue]
        A = np.multiply(1-states[:, idx], np.reshape(corr_vals[:, n_cue], (int(T/dt), 1)))*dt
        
        positions_corr[:, idx] = A
    #positions_corr = np.multiply(1-states, corr_vals)*dt
    
    # calcualte final random walk positions, starting at 0
    positions = np.vstack((np.zeros(N), np.cumsum(positions_ind + positions_corr, axis=0)))
    
    # identify individuals who reached a boundary (position >1 or <-1)
    decided_individuals = np.array([k for k in range(N) if np.isclose(abs(positions[:, k]), 1.0).any() ])
    # when was the boundery reached: 
    decision_time = np.array([np.argwhere(np.isclose(abs(positions[:, k]), 1.0))[0][0] for k in decided_individuals])
    
    # select the decisons of decided individuals
    decisons = np.round(positions[decision_time, decided_individuals])
    
    
    if return_everything: # return random walk trajectory
        for k_idx, k in enumerate(decided_individuals): 
            positions[decision_time[k_idx]:, k] = decisons[k_idx]
        dt = max(decision_time)+1
        return decisons, positions[:dt, :], corr_vals[:dt, :], ind_vals[:dt, :], states[:dt, :], dt
    
    return decisons



def make_plot(decision, positions, corr_vals, ind_vals, states, decision_time, params):

    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 3)
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.bottom'] = False

    plt.subplot(gs[0, :])
    plt.plot(positions, color="grey")
    plt.plot([-5, decision_time], [-1, -1], color="k")
    plt.plot([-5, decision_time], [0, 0], color="k")
    plt.plot([-5, decision_time], [1,1], color="k")
    
    plt.xlim(-1, decision_time)
    plt.xticks([])
    plt.ylim(-1.1, 1.1)
    plt.yticks([-1, 0, 1])
    plt.title('Decision Time = {}/{}, Fraction Decided = = {}/{},  Fraction Correct = {}/{} '.format(decision_time, params["time_steps"], len(decision), positions.shape[1], 
                                        len(decision[decision==1]),positions.shape[1]), size=14)



    cmap_ind = colors.ListedColormap(['white', 'cornflowerblue'])
    plt.subplot(gs[1, 0])
    plt.imshow(ind_vals.T.astype(int), aspect='auto', cmap=cmap_ind)
    cbar = plt.colorbar(ticks=[-1, 1])
    cbar.ax.set_yticklabels(['-1', "1"]) 
    plt.xticks([])
    plt.xlabel("time", size=14)
    plt.yticks(range(ind_vals.shape[1]))
    plt.ylabel('individuals', size=14)
    plt.title("Independent Cue \n(rI={}) ".format(params["rI"]), size=14)

    cmap_corr = colors.ListedColormap(['indianred', 'white'])
    plt.subplot(gs[1, 1])
    plt.imshow(corr_vals.T, aspect='auto', cmap=cmap_corr)
    cbar = plt.colorbar(ticks=[-1, 1])
    cbar.ax.set_yticklabels(['-1', "1"]) 
    plt.xticks([])
    plt.xlabel("time", size=14)
    plt.yticks(range(corr_vals.shape[1]))
    plt.ylabel('groups', size=14)
    plt.title("Correlated Cue \n(rC={}) ".format(params["rC"]), size=14)


    cmap_state = colors.ListedColormap(['indianred', 'cornflowerblue'])
    plt.subplot(gs[1, 2])
    plt.imshow(states.T, aspect='auto', cmap=cmap_state)
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(['corr', "ind"], size=14) 
    plt.xticks([])
    plt.xlabel("time", size=14)
    plt.yticks(range(states.shape[1]))
    plt.ylabel('individuals', size=14)
    plt.title("State \n(r10 = {}, r01 = {})".format(params['rate10'], params["rate01"]), size=14)


    plt.show()