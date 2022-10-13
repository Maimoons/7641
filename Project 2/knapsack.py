import mlrose_hiive
from base_randomization import *


if __name__ == "__main__": 
    problem = "knapsack"
    
    weights = [10, 5, 2, 8, 15]
    values = [1, 2, 3, 4, 5]
    fitness = mlrose_hiive.Knapsack(weights, values)

    parameters = {
        "restarts": 10,
        "max_attempts": 10,
        "max_iters": 1000,
        "schedule": mlrose_hiive.GeomDecay(),
        "pop_size": 200,
        "mutation_prob": 0.2
    }
        
    randomization(problem, fitness, parameters)
