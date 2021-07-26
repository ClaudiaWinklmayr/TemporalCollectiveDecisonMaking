import numpy as np



def DichotomousMarkovProcess(time_steps,rate01=1,rate10=1,dt=0.02):
    ''' Simple function to generate a single realization of a dichotomous Markov switchting process'''
    state=np.zeros(len(time_steps));
    for s in time_steps[1:]:
        rnd=np.random.uniform()
        if(state[s-1]==0):
            state[s]=0;
            if(rnd<rate01*dt):
                state[s]=1;
        else:
            state[s]=1;
            if(rnd<rate10*dt):
                state[s]=0;
        
    return state


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


def get_ind_cue_values(N, time_steps, rI):
    ''' return values of the independent cue for N individuals over all time steps '''
    return 2*(np.random.rand(time_steps, N)<rI).astype(int)-1


def get_corr_cue_values(time_steps, rC, cue_updates):
    ''' return values of the correlated cue which is updated at expoentially distributed intervalls (mean = cueUpdates)'''
    
    if cue_updates > time_steps: # in this cases the correlated cue remains constant
        print("set correlated cue fixed")
        return (2*int(np.random.uniform()<rC)-1) * np.ones((time_steps, 1))
    
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
    
    return np.reshape(values, (time_steps, 1))



def init_params(N=51, T=10, dt=0.1, init_state=[], rate01=2.0, rate10=1.0, rI=0.5, rC=0.5, cue_updates=2): 
    params={}
    params['N'] = N # number of individuals
    params['T'] = T # total integration time
    params['dt'] = dt # step size
    params['rate01'] = rate01 # rate of switching from independent to correlated cue
    params['rate10'] = rate10 # rate of switching from correlated to independent cue
    params['rI'] = rI # reliability of independent cue 
    params['rC'] = rC # reliability of correlated cue 
    params['cue_updates'] = cue_updates # averageupdate intervalls of correlated cue 
    
    if (init_state == []) or (len(init_state) != N): 
        init_state = np.ones(N)
        p_0=rate10/(rate01+rate10)
        rand = np.random.rand(N)
        init_state[rand<p_0] = -1
        
    params['init_state'] = init_state
    
    return params 



def RandomWalkDMP(params, return_positions=False):
    N = params['N']
    T = params['T']
    dt = params['dt']
    
    # draw time line of individuals' states
    states = collectiveDMP(params['init_state'],params['rate01'],params['rate10'],int(T/dt),dt)
    # draw time line of values of the independent cue 
    ind_vals = get_ind_cue_values(N, int(T/dt), params['rI'])
    # draw values of the correlated cue
    corr_vals= get_corr_cue_values(int(T/dt), params['rC'], params['cue_updates'])
    
    
    # rescale state values {-1, 1} <- {0,1}
    states= 0.5*(1+states)# 1 is independent 0 is correlated

    # claculate the individuals' positions based on the independent cue 
    positions_ind = np.multiply(states, ind_vals)*dt
    # claculate the individuals' positions based on the currelated cue 
    positions_corr = np.multiply(1-states, corr_vals)*dt
    
    # calcualte final random walk positions, starting at 0
    positions = np.vstack((np.zeros(N), np.cumsum(positions_ind + positions_corr, axis=0)))
    
    # identify individuals who reached a boundary (position >1 or <-1)
    decided_individuals = np.array([k for k in range(N) if np.isclose(abs(positions[:, k]), 1.0).any() ])
    # when was the boundery reached: 
    decision_time = np.array([np.argwhere(np.isclose(abs(positions[:, k]), 1.0))[0][0] for k in decided_individuals])
    
    # select the decisons of decided individuals
    decisons = np.round(positions[decision_time, decided_individuals])
    
    
    if return_positions: # return random walk trajectory
        for k_idx, k in enumerate(decided_individuals): 
            positions[decision_time[k_idx]:, k] = decisons[k_idx]
    
        return decisons, positions
    
    return decisons