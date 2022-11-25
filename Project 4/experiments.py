
import hiive.mdptoolbox
import hiive.mdptoolbox.mdp
import hiive.mdptoolbox.example 
import gym
from gym.envs.toy_text import frozen_lake
from openai import OpenAI_MDPToolbox 

from policy_iteration import PolicyIteration
from value_iteration import ValueIteration
from q_learn import QLearning

import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as pyplot

def run_policy_itr_experiment():   
    print("Running Policy Iteration")      		  	  		  		  		    	 		 		   		 		  
    PI = PolicyIteration(P, Reward, problem_name[p_name_i], size_name[size_i], size[size_i], env=env)	  
    PI.run_epsilon_cv()	  
    
def run_value_itr_experiment():   
    print("Running Value Iteration")      		  	  		  		  		    	 		 		   		 		        		  	  		  		  		    	 		 		   		 		  
    VI = ValueIteration(P, Reward, problem_name[p_name_i], size_name[size_i], size[size_i], env=env)	  
    VI.run_epsilon_cv()	         
 
def run_qlearn_experiment():   
    print("Running Q Learning")      		  	  		  		  		    	 		 		   		 		        		  	  		  		  		    	 		 		   		 		  
    Q = QLearning(P, Reward, problem_name[p_name_i], size_name[size_i], size[size_i], env=env)	  
    Q.run()	         

def run_all_experiments():
    run_policy_itr_experiment()
    run_value_itr_experiment()
    run_qlearn_experiment()

def plot_bar_chart(PI, VI, QL, problem_name, y_name, title):
    
    xticks = np.arange(len(size_name))

    fig, ax = pyplot.subplots()
    width = 0.3
    PI_bar = ax.bar(xticks, PI, width, label = 'Policy Iteration')
    VI_bar = ax.bar(xticks + width, VI, width, label = 'Value Iteration')
    QL_bar = ax.bar(xticks + 2*width, QL, width, label = 'Q Learning')

    ax.set_ylabel(y_name)
    ax.set_title(title+": \n Comparison of algorithms across problem sizes")
    ax.set_xticks(xticks+width)
    ax.set_xticklabels(size_name)
    plt.xticks(rotation=90)
    ax.legend()
    fig.tight_layout()
    plt.savefig("./images/"+problem_name+"_"+y_name+"_all_models", bbox_inches="tight")
    plt.clf()

def plot_experiments_forest():
    PI = [1, 2, 3] # small, medium, large
    VI = [1, 2, 3]
    QL = [1, 2, 3]   
    plot_bar_chart(PI, VI, QL, problem_name[p_name_i], "time", "Average Time")
    
    PI = [1, 2, 3] 
    VI = [1, 2, 3]
    QL = [1, 2, 3] 
    plot_bar_chart(PI, VI, QL, problem_name[p_name_i], "cum_reward", "Average Cumulative Reward")
    
    PI = [1, 2, 3] 
    VI = [1, 2, 3]
    QL = [1, 2, 3] 
    plot_bar_chart(PI, VI, QL, problem_name[p_name_i], "max_v", "Average Max Valuation")

    PI = [1, 2, 3] 
    VI = [1, 2, 3]
    QL = [1, 2, 3] 
    plot_bar_chart(PI, VI, QL, problem_name[p_name_i], "itr", "Average Number of Iterations")
    
    PI = [1, 2, 3] 
    VI = [1, 2, 3]
    QL = [1, 2, 3] 
    plot_bar_chart(PI, VI, QL, problem_name[p_name_i], "delta", "Average Delta in Valuation")


def plot_experiments_lake():
    PI = [] # small, medium, large
    VI = []
    QL = []   
    plot_bar_chart(PI, VI, QL, problem_name[p_name_i], "time", "Average Time")
    
    PI = []
    VI = []
    QL = [] 
    plot_bar_chart(PI, VI, QL, problem_name[p_name_i], "cum_reward", "Average Cumulative Reward")
    
    PI = []
    VI = []
    QL = [] 
    plot_bar_chart(PI, VI, QL, problem_name[p_name_i], "max_v", "Average Max Valuation")

    PI = []
    VI = []
    QL = [] 
    plot_bar_chart(PI, VI, QL, problem_name[p_name_i], "itr", "Average Number of Iterations")
    
    PI = []
    VI = []
    QL = [] 
    plot_bar_chart(PI, VI, QL, problem_name[p_name_i], "delta", "Average Delta in Valuation")


         
if __name__ == "__main__": 
    random.seed(10); np.random.seed(10)
    
    problem_name = ["FrozenLake", "ForestManagement"]; p_name_i = 0
    size_name = ["Small", "Medium", "Large"]; size_i = 0
    size = [4, 10, 20] #lake size
    states = [20, 200, 500] #forest states
    r1_r2 = [(10,2), (50,5), (100,15)]
    
    env = None 
    
    print("Running: " + problem_name[p_name_i] + " " + size_name[size_i])
    
    if p_name_i == 0:
        map = frozen_lake.generate_random_map(size=size[size_i], p=0.9) #p = tile frozen
        env = OpenAI_MDPToolbox("FrozenLake-v1", desc = map)
        P, Reward = env.P, env.R

    if p_name_i == 1:
        P, Reward = hiive.mdptoolbox.example.forest(S= states[size_i],
                                                    p=0.01,
                                                    r1=r1_r2[size_i][0],
                                                    r1=r1_r2[size_i][1])	#p = fire
        
    run_all_experiments()
    #plot_experiments_forest()
    
     		  	  		  		  		    	 		 		   		 		  
