import mlrose_hiive
from base_randomization import *


if __name__ == "__main__": 
    problem = "four_peaks"
    fitness = mlrose_hiive.FourPeaks()

    parameters = {
        "restarts": 10,
        "max_attempts": 10,
        "max_iters": 1000,
        "schedule": mlrose_hiive.GeomDecay(),
        "pop_size": 200,
        "mutation_prob": 0.2
    }
        
    randomization(problem, fitness, parameters)
