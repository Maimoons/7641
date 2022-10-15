import mlrose_hiive
import numpy as np
import time
from base import *

    
    
def randomization(problem, fitness_list, state_vectors, parameters, maximize, isTSP = False):
    
    def random_hill(problem_fit):        
        start_time = time.time()
        _, best_fitness, fitness_curve = mlrose_hiive.random_hill_climb(problem_fit,
                                                random_state= 10,
                                                restarts=parameters["restarts"],
                                                max_attempts=parameters["max_attempt"],
                                                max_iters=parameters["max_iters"],
                                                curve=True)
        
        time_elapsed = time.time()-start_time

        all_time_elapsed_list[0].append(time_elapsed)
        all_best_fitness_list[0].append(best_fitness)
        fitness_curve_list.append(fitness_curve[:,0])
          
        
    def simulated_annealing(problem_fit):        
        start_time = time.time()
        _, best_fitness, fitness_curve = mlrose_hiive.simulated_annealing(problem_fit,
                                                random_state= 10,
                                                max_attempts=parameters["max_attempt"],
                                                max_iters=parameters["max_iters"],
                                                schedule=parameters["schedule"],
                                                curve=True)
        time_elapsed = time.time()-start_time

        all_time_elapsed_list[1].append(time_elapsed)
        all_best_fitness_list[1].append(best_fitness)
        fitness_curve_list.append(fitness_curve[:,0])
        
    def genetic_alg(problem_fit): 
        start_time = time.time()
        _, best_fitness, fitness_curve = mlrose_hiive.genetic_alg(problem_fit,
                                                random_state= 10,
                                                max_attempts=parameters["max_attempt"],
                                                max_iters=parameters["max_iters"],
                                                pop_size=parameters["pop_size_genetic"],
                                                mutation_prob=parameters["mutation_prob"],
                                                curve=True)
        time_elapsed = time.time()-start_time

        all_time_elapsed_list[2].append(time_elapsed)
        all_best_fitness_list[2].append(best_fitness)
        fitness_curve_list.append(fitness_curve[:,0])
        
        
    def mimic(problem_fit):
        start_time = time.time()
        _, best_fitness, fitness_curve = mlrose_hiive.mimic(problem_fit,
                                                random_state= 10,
                                                max_attempts=parameters["max_attempt"],
                                                max_iters=parameters["max_iters"],
                                                pop_size=parameters["pop_size_mimic"],
                                                curve=True)
        time_elapsed = time.time()-start_time

        all_time_elapsed_list[3].append(time_elapsed)
        all_best_fitness_list[3].append(best_fitness)
        fitness_curve_list.append(fitness_curve[:,0])

        
    algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_alg', 'mimic']
    
    all_best_fitness_list = [[] for _ in range(4)]
    all_time_elapsed_list = [[] for _ in range(4)]
    
    for idx, state_vector in enumerate(state_vectors):
        fitness_curve_list = []
        fitness = fitness_list[idx]
        if isTSP:
            problem_fit = mlrose_hiive.TSPOpt(length = state_vector,fitness_fn = fitness,maximize = maximize)
        else:
            problem_fit = mlrose_hiive.DiscreteOpt(length = state_vector,fitness_fn = fitness,maximize = maximize)

        random_hill(problem_fit)
        simulated_annealing(problem_fit)
        genetic_alg(problem_fit)
        mimic(problem_fit)
        plot_all_fitness_loss(fitness_curve_list, algorithms, problem, state_vector)

    plot_all_best_fitness(all_best_fitness_list, state_vectors, algorithms, problem)
    plot_all_time_elapsed(all_time_elapsed_list, state_vectors, algorithms, problem)
    print("best_fitness:", all_best_fitness_list, "\n")
    print("time_elapsed:", all_time_elapsed_list, "\n")
       

