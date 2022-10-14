import mlrose_hiive
import numpy as np
import time
from base import *

    
    
def randomization(problem, fitness, parameters):
    
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
        fitness_curve_list.append(fitness_curve)
          
        
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
        fitness_curve_list.append(fitness_curve)
        
    def genetic_alg(problem_fit): 
        start_time = time.time()
        _, best_fitness, fitness_curve = mlrose_hiive.genetic_alg(problem_fit,
                                                random_state= 10,
                                                restarts=10,
                                                max_attempts=parameters["max_attempt"],
                                                max_iters=parameters["max_iters"],
                                                pop_size=parameters["pop_size"],
                                                mutation_prob=parameters["mutation_prob"],
                                                curve=True)
        time_elapsed = time.time()-start_time
        
        all_time_elapsed_list[2].append(time_elapsed)
        all_best_fitness_list[2].append(best_fitness)
        fitness_curve_list.append(fitness_curve)
        
        
    def mimic(problem_fit):
        start_time = time.time()
        _, best_fitness, fitness_curve = mlrose_hiive.mimic(problem_fit,
                                                random_state= 10,
                                                restarts=10,
                                                max_attempts=parameters["max_attempt"],
                                                max_iters=parameters["max_iters"],
                                                pop_size=parameters["pop_size"],
                                                fast_mimic=True,
                                                curve=True)
        time_elapsed = time.time()-start_time
        
        all_time_elapsed_list[3].append(time_elapsed)
        all_best_fitness_list[3].append(best_fitness)
        fitness_curve_list.append(fitness_curve)

    
    
    
    state_vectors = np.arange(10, 101, 10)
    
    algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_alg', 'mimic']
    
    all_best_fitness_list = [[] for _ in range(4)]
    all_time_elapsed_list = [[] for _ in range(4)]
    
    for state_vector in state_vectors:
        fitness_curve_list = []
        problem_fit = mlrose_hiive.DiscreteOpt(length = state_vector,fitness_fn = fitness,maximize = True)
        
        random_hill(problem_fit, state_vector)
        simulated_annealing(problem_fit, state_vector)
        genetic_alg(problem_fit, state_vector)
        mimic(problem_fit, state_vector)
        plot_all_fitness_loss(fitness_curve_list, algorithms, problem)

    plot_all_best_fitness(all_best_fitness_list, state_vectors, algorithms, problem)
    plot_all_time_elapsed(all_time_elapsed_list, state_vectors, algorithms, problem)
       

def randomization_CV(problem, fitness, parameters):
    
    def random_hill(problem_fit):        
        runner = mlrose_hiive.RHCRunner(problem_fit,
                                                seed= 10,
                                                restart_list=parameters["restart_list"],
                                                max_attempts=parameters["max_attempt"],
                                                iteration_list=parameters["iteration_list"],
                                                output_directory = "",
                                                generate_curves=True)
        
        start_time = time.time()
        df_run_stats, df_run_curves = runner.run()
        time_elapsed = time.time()-start_time
        
        all_time_elapsed_list[0].append(time_elapsed)
        all_best_fitness_list[0].append(best_fitness)
        fitness_curve_list.append(fitness_curve)
          
        
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
        fitness_curve_list.append(fitness_curve)
        
    def genetic_alg(problem_fit): 
        start_time = time.time()
        _, best_fitness, fitness_curve = mlrose_hiive.genetic_alg(problem_fit,
                                                random_state= 10,
                                                restarts=10,
                                                max_attempts=parameters["max_attempt"],
                                                max_iters=parameters["max_iters"],
                                                pop_size=parameters["pop_size"],
                                                mutation_prob=parameters["mutation_prob"],
                                                curve=True)
        time_elapsed = time.time()-start_time
        
        all_time_elapsed_list[2].append(time_elapsed)
        all_best_fitness_list[2].append(best_fitness)
        fitness_curve_list.append(fitness_curve)
        
        
    def mimic(problem_fit):
        start_time = time.time()
        _, best_fitness, fitness_curve = mlrose_hiive.mimic(problem_fit,
                                                random_state= 10,
                                                restarts=10,
                                                max_attempts=parameters["max_attempt"],
                                                max_iters=parameters["max_iters"],
                                                pop_size=parameters["pop_size"],
                                                fast_mimic=True,
                                                curve=True)
        time_elapsed = time.time()-start_time
        
        all_time_elapsed_list[3].append(time_elapsed)
        all_best_fitness_list[3].append(best_fitness)
        fitness_curve_list.append(fitness_curve)

    
    
    
    state_vectors = np.arange(10, 101, 10)
    
    algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_alg', 'mimic']
    
    all_best_fitness_list = [[] for _ in range(4)]
    all_time_elapsed_list = [[] for _ in range(4)]
    
    for state_vector in state_vectors:
        fitness_curve_list = []
        problem_fit = mlrose_hiive.DiscreteOpt(length = state_vector,fitness_fn = fitness,maximize = True)
        
        random_hill(problem_fit, state_vector)
        simulated_annealing(problem_fit, state_vector)
        genetic_alg(problem_fit, state_vector)
        mimic(problem_fit, state_vector)
        plot_all_fitness_loss(fitness_curve_list, algorithms, problem)

    plot_all_best_fitness(all_best_fitness_list, state_vectors, algorithms, problem)
    plot_all_time_elapsed(all_time_elapsed_list, state_vectors, algorithms, problem)