"""
This is the machinnery that runs your agent in an environment.

"""
import matplotlib.pyplot as plt
import numpy as np
import agent
import time
from sir import runSIR

class Runner:
    def __init__(self, environment, agent, training_reward_path, test_reward_path,verbose=False):
        self.environment = environment
        self.agent = agent
        self.verbose = verbose
        self.training_reward_path = training_reward_path
        self.test_reward_path = test_reward_path

    def step(self):
        observation = self.environment.observe().clone()
        action = self.agent.act(observation).copy()
        (reward, done) = self.environment.act(action)
        self.agent.reward(observation, action, reward,done)
        return (observation, action, reward, done)

    def evaluate(self,max_iter,num_episodes=1): 
        """ Start evaluation """
        print(f'\n{"-"*50}start evaluation{"-"*50}')
        episode_accumulated_rewards = np.empty((10, num_episodes))
        mode = 'test'
        g_names = []
        print(len(self.environment.graphs))
        for g_index in range(len(self.environment.graphs)-3,len(self.environment.graphs)):
            start_time = time.time()
            self.environment.reset(g_index)
            self.agent.reset(g_index) 
            for episode in range(num_episodes):
                # select other graphs
               
                self.environment.reset(g_index)
                self.agent.reset(g_index)  
                accumulated_reward = 0
                actions = []
                for i in range(1, max_iter + 1):     
                   # print(i)
                    observation = self.environment.observe().clone()
                    action = self.agent.act(observation).copy()
                    actions.append(action)
                    (_, done) = self.environment.act(action)
                    if done:     
                        actions = (set(actions).difference(set(self.environment.ini_infecteds)))
                        graph_remained = self.environment.graph_init.g.copy()                
                        graph_remained.remove_nodes_from(actions)
                        accumulated_reward = runSIR(graph_remained,self.environment.ini_infecteds,times=50,tau=1.2, gamma=1)
                        episode_accumulated_rewards[g_index-len(self.environment.graphs)+10, episode] = accumulated_reward
                        break
            end_time = time.time()
            print('runtime for one graph is: ', end_time-start_time) 
        ave_cummulative_rewards = np.mean(episode_accumulated_rewards, axis=1)
        ave_cummulative_reward = np.mean(ave_cummulative_rewards)

        print('average cummulative reward is: ', ave_cummulative_reward)
        print(f'{"-"*50}end evaluation{"-"*50}')
        print(' ')
        return ave_cummulative_reward    
    
    
    

    def loop(self, games,nbr_epoch, max_iter,savedname):

        cumul_reward = 0.0
        for epoch_ in range(nbr_epoch):
            start = time.time()
            list_cumul_reward=[]
            list_cumul_reward_test = []
            print(" -> epoch : "+str(epoch_))
            #for g in range(1, games + 1):
            for g in range(games):
                print(" -> games : "+str(g))
                self.environment.reset(g)
                self.agent.reset(g)
                cumul_reward = 0.0
                for i in range(1, max_iter + 1):     
                    (obs, act, rew, done) = self.step()
                    cumul_reward += rew
                    if self.verbose:
                        if done:
                            print(" ->    Terminal event: cumulative rewards = {}".format(cumul_reward))
                            list_cumul_reward.append(cumul_reward)
                    if done:
                        break                       
#             if epoch_ % 10 == 0:
#                 episode_accumulated_reward = self.evaluate(max_iter)  
#                 list_cumul_reward_test.append(episode_accumulated_reward)
                           
                          
            print(f'time for one epoch: {time.time()-start}')
            if self.verbose:
                print(" <=> Finished game number: {} <=>".format(g))
                print("")
                
            with open(self.training_reward_path, "ab") as f:
                np.savetxt(f, list_cumul_reward, fmt='%.4f',delimiter=',')
#             with open(self.test_reward_path, "ab") as f:
#                 np.savetxt(f, list_cumul_reward_test, fmt='%.4f',delimiter=',')

            if epoch_ % 100 == 0:
                self.agent.save_model(str(epoch_)+savedname)
            

        return cumul_reward

def iter_or_loopcall(o, count):
    if callable(o):
        return [ o() for _ in range(count) ]
    else:
        # must be iterable
        return list(iter(o))

class BatchRunner:
    """
    Runs several instances of the same RL problem in parallel
    and aggregates the results.
    """

    def __init__(self, env_maker, agent_maker, count, verbose=False):
        self.environments = iter_or_loopcall(env_maker, count)
        self.agents = iter_or_loopcall(agent_maker, count)
        assert(len(self.agents) == len(self.environments))
        self.verbose = verbose
        self.ended = [ False for _ in self.environments ]

    def game(self, max_iter):
        rewards = []
        for (agent, env) in zip(self.agents, self.environments):
            env.reset()
            agent.reset()
            game_reward = 0
            for i in range(1, max_iter+1):
                observation = env.observe()
                action = agent.act(observation)
                (reward, stop) = env.act(action)
                agent.reward(observation, action, reward)
                game_reward += reward
                if stop :
                    break
            rewards.append(game_reward)
        return sum(rewards)/len(rewards)

    def loop(self, games,nb_epoch, max_iter):
        cum_avg_reward = 0.0
        for epoch in range(nb_epoch):
            for g in range(1, games+1):
                avg_reward = self.game(max_iter)
                cum_avg_reward += avg_reward
                if self.verbose:
                    print("Simulation game {}:".format(g))
                    print(" ->            average reward: {}".format(avg_reward))
                    print(" -> cumulative average reward: {}".format(cum_avg_reward))
        return cum_avg_reward
