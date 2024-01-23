"""
This is the machinnery that runs your agent in an environment.

"""
import matplotlib.pyplot as plt
import numpy as np
import agent
import time
from sir import runSIR

class Runner:
    def __init__(self, environment, agent, times,verbose=False):
        self.environment = environment
        self.agent = agent
        self.verbose = verbose
        self.times = times


    def evaluate(self,max_iter,seeds): 

        print(f'\n{"-"*50}start evaluation{"-"*50}')
        results = {}
        #episode_accumulated_rewards = np.empty((5, self.times))
    
        for g_index in range(5):
            start_time = time.time()

            self.environment.reset(g_index,seeds)
            self.agent.reset(g_index,seeds)  
            accumulated_reward = 0
            actions = []
            for i in range(1, max_iter + 1):     
                observation = self.environment.observe().clone()
                action = self.agent.act(observation).copy()
                actions.append(action)
                (_, done) = self.environment.act(action)
                if done:     
                    actions = (set(actions).difference(set(self.environment.ini_infecteds)))
                    graph_remained = self.environment.graph_init.g.copy()                
                    graph_remained.remove_nodes_from(actions)
                    accumulated_reward = runSIR(graph_remained,self.environment.ini_infecteds,times=50,tau=1.2, gamma=1)
                    #episode_accumulated_rewards[g_index, episode] = accumulated_reward
                    break  
            results[g_index] = actions
            end_time = time.time()
            print('runtime for one graph is: ', end_time-start_time) 
        #ave_cummulative_rewards = np.mean(episode_accumulated_rewards, axis=1)
        #ave_cummulative_reward = np.mean(ave_cummulative_rewards)
        return results,accumulated_reward
    
    
    
    
