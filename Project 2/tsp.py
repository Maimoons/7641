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
    problem = "tsp"
    
    state_vectors = [i for i in range(10, 101, 10)]
    fitness_list = []

    for vector in state_vectors:
        np.random.seed(vector)
        coords_list = [(np.random.uniform(-10*vector,10*vector), np.random.uniform(-10*vector,10*vector)) for _ in range(vector)]
        fitness = mlrose_hiive.TravellingSales(coords = coords_list)
        fitness_list.append(fitness)
        
    parameters = {
        "restarts": 10,
        "max_attempt": 100,
        "max_iters": 1500,
        "schedule": mlrose_hiive.GeomDecay(),
        "pop_size_genetic": 200,
        "pop_size_mimic": 200,
        "mutation_prob": 0.5,
    }
    
    randomization(problem, fitness_list, state_vectors, parameters, maximize = False, isTSP = True)
