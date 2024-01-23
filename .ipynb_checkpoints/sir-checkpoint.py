import EoN


def runSIR(G,ini_infecteds,times=1000,tau=1.5, gamma=1 ):
    iterations = times
    Isum = 0
    for counter in range(iterations):
#         newI = EoN.get_infected_nodes(G, tau, gamma, initial_infecteds=ini_infecteds)
#         Isum += len(newI)
        t, S, I, R = EoN.fast_SIR(G, tau, gamma,initial_infecteds = ini_infecteds)
        Isum += S[-1]
    Iave = Isum / float(iterations)
    
    return (G.number_of_nodes() - Iave)








def runIC (G, S, p=0.1 ):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial list of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''
    
    T = deepcopy(S) 
    for u in T: 
         for v in G[u]: 
            w = 1
            if v not in T and random.random() < 1 - (1-p)**w:
                T.append(v)
    return T

def runIC_repeat(G, S, p=0.01, sample=1000):
    infl_list = []
    for i in range(sample):
        T = runIC(G, S, p=p)
        influence = len(T)
        infl_list.append(influence)
    infl_mean = np.mean(infl_list)
    infl_std = np.std(infl_list)

    return infl_mean, infl_std 