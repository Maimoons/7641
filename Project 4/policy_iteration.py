import hiive.mdptoolbox
import hiive.mdptoolbox.mdp
import hiive.mdptoolbox.example 
import gym
from gym.envs.toy_text import frozen_lake
from openai import OpenAI_MDPToolbox

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects

from custom_PolicyIteration import custom_PolicyIteration
   
class PolicyIteration():
    
    def __init__(self, transitions, rewards, problem_name, size_name, size = None, env = None) -> None:
        self.env = env
        self.mdp_P = transitions
        self.mdp_Rewards = rewards
        self.problem_name = problem_name
        self.size = size
        self.size_name = size_name
        self.gamma_list = [0.1, 0.2] #[0.4, 0.6, 0.9, 0.95, 0.99, 1]
        self.epsilon_list = [0.3, 0.4] #[0.05, 0.15, 0.25, 0.5, 0.75, 0.95]
        self.max_iter = 5 #10000
        self.best_epsilon = 0.0001
        self.best_policy = None
        
        self.keys = ["Reward", "Delta", "Time", "Max V", "Cum Reward", "Iterations", "Threshold"]
        self.epsilon_gamma_stat_dict = {g: {k:{e:[] for e in self.epsilon_list} for k in self.keys} for g in self.gamma_list}
        
        self.thresholds = []
     
    def run_epsilon_cv(self):
        print("Running Epsilon tuning")
        self.run_cv(self.epsilon_list, "Epsilon", self.epsilon_gamma_stat_dict)
        
    def run_cv(self, param_list, param_name, param_stat_dict):
        best_cum_reward = 0
                        
        avg_stat_dict = {g: {k:[] for k in param_list} for g in self.keys} 

        for _,param in enumerate(param_list):
            for _,gamma in enumerate(self.gamma_list):
                PI_mdp = custom_PolicyIteration\
                    (self.mdp_P, self.mdp_Rewards, epsilon = param, gamma=gamma, max_iter=self.max_iter)
                
                PI_mdp.run()
                #PI_mdp.verbose = True
                
                threshold = PI_mdp.epsilon * (1 - PI_mdp.gamma) / PI_mdp.gamma
                policy = PI_mdp.policy
                
                stats = PI_mdp.run_stats            
                cum_rewards = stats[-1]["Mean V"]
                
                if (cum_rewards > best_cum_reward):
                    best_params = (param,gamma)
                    self.best_policy = policy
                        
                reward, delta, time, max_v, cum_reward = self.calculate_stats(stats)
                self.add_stats(param_stat_dict, param, gamma, reward, delta, time, max_v, cum_reward)                        
                self.add_averages(threshold, param, avg_stat_dict, reward, delta, time, max_v, cum_reward)
            
        
        self.plot_iteration(param_stat_dict, best_params, param_name)  
        self.plot_averages(avg_stat_dict, "Threshold", param_list)

        
        if self.env:
            policy_array = np.array(self.best_policy).reshape(self.size,self.size)
            self.plot_policy(policy_array, self.env.unwrapped.desc, self.env.colors(), self.env.directions(),\
                "Best Policy; Gamma, %s: (%s , %s)" % (param_name, str(best_params[1]),str(best_params[0])), "policy", param_name)
        else:
            print("Best Policy; Gamma, %s: (%s , %s)" % (param_name, str(best_params[1]), str(best_params[0])))
            print(self.best_policy)
        
        best_param, best_gamma = best_params
        print("Best Gamma,"+param_name, best_gamma, best_param)
        print("Avg Time", np.mean(avg_stat_dict["Reward"][best_param]))
        print("Avg Cum Reward", np.mean(avg_stat_dict["Cum Reward"][best_param]))
        print("Avg Max V", np.mean(avg_stat_dict["Max V"][best_param]))
        print("Avg Itr", np.mean(avg_stat_dict["Iterations"][best_param]))
        print("Avg Delta", np.mean(avg_stat_dict["Delta"][best_param]))
        
    
    def calculate_stats(self, stats_list):
        reward, delta, time, max_v, cum_reward = [[] for _ in range(5)]
        
        for dict in stats_list:
            reward.append(dict["Reward"])
            delta.append(dict["Error"])
            time.append(dict["Time"])
            max_v.append(dict["Max V"])
            cum_reward.append(dict["Mean V"])
         
        return reward, delta, time, max_v, cum_reward

    def add_stats(self, param_stat_dict, param, gamma, reward, delta, time, max_v, cum_reward):
        old_r = param_stat_dict[gamma]["Reward"] 
        old_r.update({param: reward})
        param_stat_dict[gamma].update({"Reward": old_r})
        
        old_d = param_stat_dict[gamma]["Delta"] 
        old_d.update({param: delta})
        param_stat_dict[gamma].update({"Delta": old_d})
        
        old_t = param_stat_dict[gamma]["Time"] 
        old_t.update({param: time})
        param_stat_dict[gamma].update({"Time": old_t})
        
        old_mv = param_stat_dict[gamma]["Max V"] 
        old_mv.update({param: max_v})
        param_stat_dict[gamma].update({"Max V": old_mv})
        
        old_cr = param_stat_dict[gamma]["Cum Reward"] 
        old_cr.update({param: cum_reward})
        param_stat_dict[gamma].update({"Cum Reward": old_cr})

        #print(gamma, param, reward, delta, time, max_v, cum_reward, "\n", param_stat_dict, "\n")

    def add_averages(self, threshold, param, avg_stat_dict, reward, delta, time, max_v, cum_reward):  
        new_val = avg_stat_dict["Reward"][param]+[reward[-1]]
        old_r = avg_stat_dict["Reward"]
        old_r.update({param:new_val})
        avg_stat_dict.update({"Reward": old_r})
        
        new_val = avg_stat_dict["Delta"][param]+[delta[-1]] 
        old_d = avg_stat_dict["Delta"]
        old_d.update({param:new_val})
        avg_stat_dict.update({"Delta": old_d})
        
        new_val = avg_stat_dict["Time"][param]+[time[-1]]
        old_t = avg_stat_dict["Time"]
        old_t.update({param:new_val})
        avg_stat_dict.update({"Time": old_t})
        
        new_val = avg_stat_dict["Max V"][param]+[max_v[-1]]
        old_mv = avg_stat_dict["Max V"]
        old_mv.update({param:new_val})
        avg_stat_dict.update({"Max V": old_mv})
        
        new_val = avg_stat_dict["Cum Reward"][param]+[cum_reward[-1]]
        old_cr = avg_stat_dict["Cum Reward"]
        old_cr.update({param:new_val})
        avg_stat_dict.update({"Cum Reward": old_cr})
        
        new_val = avg_stat_dict["Iterations"][param]+[len(reward)]
        old_i = avg_stat_dict["Iterations"]
        old_i.update({param:new_val})
        avg_stat_dict.update({"Iterations": old_i})
        
        new_val = avg_stat_dict["Threshold"][param]+[threshold]
        old_t = avg_stat_dict["Threshold"]
        old_t.update({param:new_val})
        avg_stat_dict.update({"Threshold": old_t})
        
        
    def plot_iteration(self, param_stat_dict, best_param, param_name):
        _, gamma = best_param
        self.plot_iteration_graph(param_stat_dict[gamma]["Reward"], "Reward", "Reward across iterations", "reward", gamma, param_name)
        self.plot_iteration_graph(param_stat_dict[gamma]["Delta"], "Policy Valuation Delta", "Delta across iterations", "delta", gamma, param_name)
        self.plot_iteration_graph(param_stat_dict[gamma]["Time"], "Time", "Clock Time across iterations", "time", gamma, param_name)
        self.plot_iteration_graph(param_stat_dict[gamma]["Max V"], "Max Policy Valuation", "Max Valuation across iterations", "max_v", gamma, param_name)
        self.plot_iteration_graph(param_stat_dict[gamma]["Cum Reward"], "Cumulative Reward", "Cumulative Reward across iteration", "cum_reward", gamma, param_name)

    def plot_averages(self, avg_stat_dict, param_name, param_list):
        #x = param_list #epsilon
        x = np.mean(list(avg_stat_dict["Threshold"].values()), axis = 1) #threshold
        self.plot_averages_graph(list(avg_stat_dict["Time"].values()), "Average Time", x, param_name, "Convergence: Average Time to converge", "avg_time")
        self.plot_averages_graph(list(avg_stat_dict["Cum Reward"].values()), "Average Cumulative Reward", x, param_name, "Convergence: Average Cumulative Reward", "avg_cum_reward", list(avg_stat_dict["Max V"].values()))
        self.plot_averages_graph(list(avg_stat_dict["Iterations"].values()), "Number of iterations", x, param_name, "Convergence: Number of iterations to converge", "avg_num_itrs")
        self.plot_averages_graph(list(avg_stat_dict["Delta"].values()), "Delta in Valuation", x, param_name, "Convergence: Delta at convergence", "avg_deltas")
        self.plot_averages_graph(list(avg_stat_dict["Reward"].values()), "Final Reward", x, param_name, "Convergence: Reward convergence", "avg_reward")

    
    def plot_iteration_graph(self, y_dict, y_name, title, file_name, gamma, param_name):
        plt.title(title + " best gamma: "+str(gamma))

        y_values = list(y_dict.values())
        y_keys = list(y_dict.keys())
        #maxLength = max(len(x) for x in y_values )
        
        plt.xlabel('Iteration #')
        plt.ylabel(y_name)
        for i, y in enumerate(y_values):
            x = np.arange(1, len(y)+1)
            plt.plot(x, y, label = param_name+": "+ str(y_keys[i]))
        
        plt.legend(loc = 'best')        
        plt.savefig('/'.join(['./images','PolicyIteration',self.problem_name,self.size_name, param_name, file_name]), bbox_inches="tight")
        plt.clf()
    
    def plot_averages_graph(self, y, y_name, param_range, param_name, title, file_name, max_v = None, fill = True):
        y_mean = np.mean(y, axis = 1)
        y_std = np.std(y, axis = 1)
        plt.plot(param_range, y_mean, color = 'b', label = "Average")
        plt.xticks(param_range) 
        
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
        plt.savefig('/'.join(['./images','PolicyIteration',self.problem_name,self.size_name,param_name,file_name]), bbox_inches="tight")
        plt.clf()  
        print(y_name, "Mean", y_mean, "\n")
  
    def plot_policy(self, policy, map_desc, color_map, direction_map, title, file_name, param_name):
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
        plt.savefig('/'.join(['./images','PolicyIteration',self.problem_name,self.size_name, param_name, file_name]), bbox_inches="tight")
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
    PI.run_epsilon_cv()	   		  	  		  		  		    	 		 		   		 		  
