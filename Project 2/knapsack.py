import mlrose_hiive
from base_randomization import *

def get_cv():
        cv_parameters = {
        "max_attempts": 10,
        "iteration_list": [2000],
        "restart_list": [10],
        "schedule": mlrose_hiive.GeomDecay(),
        "pop_size": 200,
        "mutation_prob": 0.2
    }
        
if __name__ == "__main__": 
    problem = "knapsack"
    
    state_vectors = [i for i in range(10, 101, 10)]
    fitness_list = []
    
    for vector in state_vectors:
        weights = np.random.uniform(0.1,1, vector)
        values = np.random.uniform(1, 100, vector)
        fitness = mlrose_hiive.Knapsack(weights, values)
        fitness_list.append(fitness)
        
    parameters = {
        "restarts": 10,
        "max_attempt": 15,
        "max_iters": 1500,
        "schedule": mlrose_hiive.GeomDecay(),
        "pop_size_genetic": 10,
        "pop_size_mimic": 500,
        "mutation_prob": 0.1,
    }
     
    randomization(problem, fitness_list, state_vectors, parameters, maximize = True)
    