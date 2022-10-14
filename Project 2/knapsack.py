import mlrose_hiive
from base_randomization import *


if __name__ == "__main__": 
    problem = "knapsack"
    
    weights = [10, 5, 2, 8, 15]
    values = [1, 2, 3, 4, 5]
    fitness = mlrose_hiive.Knapsack(weights, values)

    parameters = {
        "max_attempts": 10,
        "iteration_list": [2000],
        "restart_list": [10],
        "schedule": mlrose_hiive.GeomDecay(),
        "pop_size": 200,
        "mutation_prob": 0.2
    }
        
    randomization(problem, fitness, parameters)
