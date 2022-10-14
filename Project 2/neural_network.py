import mlrose_hiive
import numpy as np
import time
from base import *


def random_hill(model_name):
    algorithm_name = 'random_hill_climb'
    
    parameters = {
        "learning_rate": np.logspace(-4,1,5),
        "restarts": np.arange(2,11,2),
    }
    
    classifier = mlrose_hiive.NeuralNetwork(hidden_nodes= [15],
                                            random_state= 10,
                                            activation='relu',
                                            algorithm= algorithm_name,
                                            max_iters = 1500,
                                            max_attempts= 150,
                                            early_stopping=True,
                                            is_classifier=True,
                                            curve=True)
    
    train_NN(model_name, parameters, classifier, algorithm_name)
    test_NN(model_name, algorithm_name)
    
def simulated_annealing(model_name):
    algorithm_name = 'simulated_annealing'
    
    parameters = {
        "learning_rate": np.logspace(-4,1,5),
        "schedule": [mlrose_hiive.GeomDecay(), mlrose_hiive.ExpDecay(), mlrose_hiive.ArithDecay()]
    }
    
    classifier = mlrose_hiive.NeuralNetwork(hidden_nodes= [15],
                                            random_state= 10,
                                            activation='relu',
                                            algorithm= algorithm_name,
                                            max_iters = 1500,
                                            max_attempts= 150,
                                            early_stopping=True,
                                            is_classifier=True,
                                            curve=True)
    
    train_NN(model_name, parameters, classifier, algorithm_name)
    test_NN(model_name, algorithm_name)
    
    
def genetic_alg(model_name):
    algorithm_name = "genetic_alg"
    
    parameters = {
        "learning_rate": [0.5],
        "pop_size": [i for i in range(100, 600, 100)],
        "mutation_prob": np.arange(0.1, 0.6, 0.1)
    }
    
    classifier = mlrose_hiive.NeuralNetwork(hidden_nodes= [15],
                                            random_state= 10,
                                            activation='relu',
                                            algorithm= algorithm_name,
                                            max_iters = 1500,
                                            max_attempts= 150,
                                            early_stopping=True,
                                            is_classifier=True,
                                            curve=True)
    
    train_NN(model_name, parameters, classifier, algorithm_name)
    test_NN(model_name, algorithm_name)
    
    
def train_NN(model_name, parameters, classifier, algorithm_name):
    model_name = model_name.format(algorithm_name)
    
    train_sizes =  np.linspace(0.1,1.0,10) 

    grid = GridSearchCV(classifier,
                        param_grid = parameters,
                        cv = 5,
                        verbose = True,
                        error_score='raise')
    grid.fit(x_train, y_train)
    
    best_classifier = grid.best_estimator_
    save_model(best_classifier, model_name)
    print(("Best Parameters: {0} \n").format(grid.best_params_))
                
    train_score, test_score, _ = learning(best_classifier, train_sizes, x_train, y_train)
    plot_train_val_curve(train_score, test_score, train_sizes, "Ratio of Train Sizes", "{0}/train_size".format(algorithm_name), dataset[dataset_idx])
    plot_fitness_loss(best_classifier, "{0}/fitness".format(algorithm_name), dataset[dataset_idx], grid.best_params_)
    print("Fitness", best_classifier.fitness_curve)
    get_train_time(best_classifier, x_train, y_train)

def test_NN(model_name, algorithm_name):
    model_name = model_name.format(algorithm_name)

    start_time = time.time()
    test(model_name, x_train, y_train, x_test, y_test, classes[dataset_idx], dataset[dataset_idx], "{0}/nn".format(algorithm_name))
    end_time = time.time()
    time_to_test = end_time - start_time
    print(("Time to Test: {0} \n").format(time_to_test))
 
       
if __name__ == "__main__": 
    dataset_idx = 1
    
    x_train, y_train, x_test, y_test = load_dataset_0() if dataset_idx == 0 else load_dataset_1()  
    dataset = ["neural_network/bc/", "neural_network/titanic/"]
    classes = [["Malignant", "Benign"], ["Not Survived","Survived"]]
    model_names = ["./models/neural_network/{0}/neuralnet_bc.pkl", "./models/neural_network/{0}/neuralnet_titanic.pkl"]

    oparameters = {
        "learning_rate": np.logspace(-4,1,1),
    }
    
    random_hill(model_names[dataset_idx])
    simulated_annealing(model_names[dataset_idx])
    genetic_alg(model_names[dataset_idx])

