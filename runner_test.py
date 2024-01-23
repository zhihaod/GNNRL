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


    def evaluate(self,max_iter,num=5): 

        print(f'\n{"-"*50}start evaluation{"-"*50}')
        results = {}
        episode_accumulated_rewards = np.empty((num, self.times))
    
        for g_index in range(num):
            start_time = time.time()
            self.environment.reset(g_index)
            self.agent.reset(g_index)
            node_count = {}
            for i in range(self.environment.graph_init.nodes()):
                node_count[i] = 0
            
            for episode in range(self.times): 
                self.environment.reset(g_index)
                self.agent.reset(g_index)  
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
                        episode_accumulated_rewards[g_index, episode] = accumulated_reward
                        break
                for action in actions:
                    node_count[action] += 1
                    
            actions_ = []
            for i in range(int(self.environment.ratio*self.environment.graph_init.nodes())):
                max_key = max(node_count, key=lambda key: node_count[key])
                actions_.append(max_key)
                del node_count[max_key]              
            results[g_index] = actions_    
            end_time = time.time()
            print('runtime for one graph is: ', end_time-start_time) 
        ave_cummulative_rewards = np.mean(episode_accumulated_rewards, axis=1)
        ave_cummulative_reward = np.mean(ave_cummulative_rewards)
        return results,ave_cummulative_reward
    
    
    
    
