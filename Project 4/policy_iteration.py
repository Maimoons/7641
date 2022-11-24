import hiive.mdptoolbox
import hiive.mdptoolbox.mdp
import hiive.mdptoolbox.example 
import gym
from gym.envs.toy_text import frozen_lake
from openai import OpenAI_MDPToolbox

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
     
class PolicyIteration():
    
    def __init__(self, transitions, rewards, problem_name, size_name, size = None, env = None) -> None:
        self.env = env
        self.mdp_P = transitions
        self.mdp_Rewards = rewards
        self.problem_name = problem_name
        self.size = size
        self.size_name = size_name
        self.gamma_list = [0.4, 0.3] #[0.4, 0.6, 0.9, 0.95, 0.99, 1]
        self.max_iter = 5 #10000
        self.best_epsilon = 0.0001
        self.best_policy = None
        self.best_gamma = None
        
        self.avg_time = []
        self.avg_cum_reward = []
        self.avg_max_v = []
        self.num_itrs = []
        self.deltas = []
        self.thresholds = []
         
    def run_cv(self):
        best_cum_reward = 0
        best_idx = 0
        best_stats = None
        best_threshold = None
        
        for i,gamma in enumerate(self.gamma_list):
            PI_mdp = hiive.mdptoolbox.mdp.PolicyIteration\
                (self.mdp_P, self.mdp_Rewards, gamma=gamma, max_iter=self.max_iter)
            
            PI_mdp.run()
            #PI_mdp.verbose = True
            
            threshold = self.best_epsilon * (1 - PI_mdp.gamma) / PI_mdp.gamma
            policy = PI_mdp.policy
            stats = PI_mdp.run_stats
            
            cum_rewards = stats[-1]["Mean V"]
            if (cum_rewards > best_cum_reward):
                self.best_gamma = gamma
                self.best_policy = policy
                best_idx = i
                best_stats = stats
                best_threshold = threshold
                    
            reward, delta, time, max_v, cum_reward = self.calculate_stats(stats)
            self.add_averages(threshold, reward, delta, time, max_v, cum_reward)
            
        
        reward, delta, time, max_v, cum_reward = self.calculate_stats(best_stats)  
        self.plot_iteration(best_idx, reward, delta, time, max_v, cum_reward)  
        self.plot_averages()
        
        if self.env:
            policy_array = np.array(self.best_policy).reshape(self.size,self.size)
            self.plot_policy(policy_array, self.env.unwrapped.desc, self.env.colors(), self.env.directions(),\
                "Best Policy; gamma, epsilon: ("+str(self.gamma_list[best_idx])+","+str(self.best_epsilon)+")", "policy")
        else:
            print("Policy")
            print(self.best_policy)
        
        print(self.best_policy)
        print("Best Gamma", self.gamma_list[best_idx])
        print("Best Threhold", best_threshold)
        print("Avg Time", np.mean(self.avg_time[best_idx]))
        print("Avg Cum Reward", np.mean(self.avg_cum_reward[best_idx]))
        print("Avg Itr", self.num_itrs[best_idx])
        print("Avg Delta", self.deltas[best_idx])
    
    def calculate_stats(self, stats_list):
        reward, delta, time, max_v, cum_reward = [[] for _ in range(5)]
        
        for dict in stats_list:
            reward.append(dict["Reward"])
            delta.append(dict["Error"])
            time.append(dict["Time"])
            max_v.append(dict["Max V"])
            cum_reward.append(dict["Mean V"])
         
        return reward, delta, time, max_v, cum_reward

    def add_averages(self, threshold, reward, delta, time, max_v, cum_reward):
        self.avg_time.append(time)
        self.avg_cum_reward.append(cum_reward)
        self.avg_max_v.append(max_v)
        self.num_itrs.append([len(reward)])
        self.deltas.append([delta[-1]]) # delta when converged
        self.thresholds.append(threshold)
        
        
    def plot_iteration(self, i, reward, delta, time, max_v, cum_reward):
        self.plot_iteration_graph(i, reward, "Reward", "Reward across iterations", "reward")
        self.plot_iteration_graph(i, delta, "Policy Valuation Delta", "Delta across iterations", "delta")
        self.plot_iteration_graph(i, time, "Time", "Clock Time across iterations", "time")
        self.plot_iteration_graph(i, max_v, "Max Policy Valuation", "Max Valuation across iterations", "max_v")
        self.plot_iteration_graph(i, cum_reward, "Cumulative Reward", "Cumulative Reward across iteration", "cum_reward", max_v = max_v)

    def plot_averages(self):
        self.plot_averages_graph(self.avg_time, "Average Time", self.thresholds, "Threshold", "Convergence: Average Time to converge", "th_avg_time")
        self.plot_averages_graph(self.avg_cum_reward, "Average Cumulative Reward", self.thresholds, "Threshold", "Convergence: Average Cumulative Reward", "th_avg_cum_reward", self.avg_max_v)
        self.plot_averages_graph(self.num_itrs, "Number of iterations", self.thresholds, "Threshold", "Convergence: Number of iterations to converge", "th_num_itrs", fill = False)
        self.plot_averages_graph(self.deltas, "Delta in Valuation", self.thresholds, "Threshold", "Convergence: Delta at convergence", "th_deltas", fill = True)

    def plot_iteration_graph(self, i, y, y_name, title, file_name, param_name = "gamma", max_v = None):
        plt.title(title + " gamma, epsilon: ("+str(self.gamma_list[i])+","+str(self.best_epsilon)+")")

        plt.xlabel('Iteration #')
        plt.ylabel(y_name)
        x = np.arange(1, len(y)+1)
        plt.plot(x, y, label = "Average "+y_name)
        
        if max_v:
            plt.plot(x, max_v,'r--', label = "Max "+y_name)    
            plt.legend(loc = 'best') 
                
        plt.savefig('/'.join(['./images','PolicyIteration',self.problem_name,self.size_name, str(i)+'_'+file_name]))
        plt.clf()
       
    def plot_averages_graph(self, y, y_name, param_range, param_name, title, file_name, max_v = None, fill = True):
        y_mean = np.mean(y, axis = 1)
        y_std = np.std(y, axis = 1)
        plt.plot(param_range, y_mean, color = 'b', label = "Average")
        if fill:
            plt.fill_between(
            param_range,
            y_mean - y_std,
            y_mean + y_std,
            alpha=0.2,
            color="b",
            lw=2,)
            
        if max_v:
            y_max = np.max(max_v, axis = 1) 
            plt.plot(param_range, y_max,'r--', label = "Max")
            plt.legend(loc = 'best')  
        
        plt.title(title)
        plt.xlabel(param_name)
        plt.ylabel(y_name)
        plt.savefig('/'.join(['./images','PolicyIteration',self.problem_name,self.size_name,file_name]))
        plt.clf()  
        print(y_name, "Mean", y_mean, "\n")
  
    def plot_policy(self, policy, map_desc, color_map, direction_map, title, file_name):
        fig = plt.figure()
        ax = fig.add_subplot(111, xlim=(0, policy.shape[1]), ylim=(0, policy.shape[0]))
        
        '''font_size = 'x-large'
        if policy.shape[1] > 16:
            font_size = 'small'''
            
        rows = policy.shape[0]
        cols = policy.shape[1]
        
        for i in range(rows):
            for x in range(cols):
                y = policy.shape[0] - i - 1
                p = plt.Rectangle([x, y], 1, 1)
                
                p.set_facecolor(color_map[map_desc[i, x]])
                ax.add_patch(p)

                text = ax.text(x+0.5, y+0.5, direction_map[policy[i, x]], weight='bold',
                            horizontalalignment='center', verticalalignment='center', color='black')
                            
                text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                    path_effects.Normal()])

        plt.tight_layout()
        plt.title(title)
        plt.axis('off')
        plt.xlim((0, policy.shape[1]))
        plt.ylim((0, policy.shape[0]))
        plt.savefig('/'.join(['./images','PolicyIteration',self.problem_name,self.size_name,file_name]))
        plt.clf()
        
            
if __name__ == "__main__":  
    problem_name = "FrozenLake"
    size_name = "small"
    
    P, Reward = hiive.mdptoolbox.example.forest(S= 2, p=0.01)	
    
    map = frozen_lake.generate_random_map(size=4, p=0.9)
    #env = OpenAI_MDPToolbox("FrozenLake-v1", desc = map)
    #P, Reward = env.P, env.R
    ##env = gym.make("FrozenLake-v1", desc=custom_map)
    #P, Reward = hiive.mdptoolbox.openai("FrozenLake", desc = map)
    
       		  	  		  		  		    	 		 		   		 		  
    PI = PolicyIteration(P, Reward, problem_name, size_name, 4, env=None)	  
    PI.run_cv()	   		  	  		  		  		    	 		 		   		 		  
