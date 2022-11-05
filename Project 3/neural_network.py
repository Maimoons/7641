from base import *
from sklearn.neural_network import MLPClassifier
import warnings

class NeuralNetwork():
    def __init__(self, x_train, y_train, x_test, y_test, dataset_idx, verbose = 0, folder = "original/"):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.dataset_idx = dataset_idx
        self.dataset = ["bc/", "titanic/"]
        self.classes = [["Malignant", "Benign"], ["Not Survived","Survived"]]
        self.folder = "neural_network/"+folder
        self.verbose = verbose
        
    def train_ANN(self, model_name):  
        warnings.filterwarnings("ignore")  
        classifier = MLPClassifier(random_state= 10, max_iter = 1000)
        
        parameters = {
            "solver": ['lbfgs', 'adam'],
            "hidden_layer_sizes": np.arange(10, 100, step=10),
            "activation": ['tanh', 'relu'],
            "learning_rate_init": np.logspace(-4,1,10),
        }
        train_sizes =  np.linspace(0.1,1.0,10)
        
        train_score, test_score = cv_validation(classifier, "hidden_layer_sizes", parameters["hidden_layer_sizes"], self.x_train, self.y_train)
        plot_train_val_curve(train_score, test_score, parameters["hidden_layer_sizes"], "Hidden units",self.folder+"hidden_layer_sizes", self.dataset[self.dataset_idx])

        train_score, test_score = cv_validation(classifier, "learning_rate_init", parameters["learning_rate_init"], self.x_train, self.y_train)
        plot_train_val_curve(train_score, test_score, parameters["learning_rate_init"], "Learning Rate", self.folder+"learning_rate_init", self.dataset[self.dataset_idx])

        grid = GridSearchCV(classifier,
                            param_grid = parameters,
                            cv = 5,
                            verbose = True)
        grid.fit(self.x_train, self.y_train)
        
        final_model = grid.best_estimator_
        save_model(final_model, model_name)
        debug(("Best Parameters: {0} \n").format(grid.best_params_))
        
        best_classifier = MLPClassifier(random_state= 10,
                                        solver = grid.best_params_["solver"],
                                        hidden_layer_sizes = grid.best_params_["hidden_layer_sizes"],
                                        activation = grid.best_params_["activation"],
                                        learning_rate_init = grid.best_params_["learning_rate_init"],
                                        max_iter= 500)
        
        train_score, test_score, _ = learning(best_classifier, train_sizes, self.x_train, self.y_train)
        plot_train_val_curve(train_score, test_score, train_sizes, "Ratio of Train Sizes", self.folder+"train_size", self.dataset[self.dataset_idx])

        self.run_epochs(grid.best_params_, 300)
        get_train_time(best_classifier, self.x_train, self.y_train)
        
    def test_ANN(self, model_name):
        start_time = time.time()
        test(model_name, self.x_train, self.y_train, self.x_test, self.y_test, self.classes[self.dataset_idx], self.dataset[self.dataset_idx], "nn", verbose = True)
        end_time = time.time()
        time_to_test = end_time - start_time
        debug(("Time to Test: {0} \n").format(time_to_test))
                    
    def run_epochs(self, best_params_, epochs):
        scores_train = []
        scores_val = []
        
        x_train_split, x_val, y_train_split, y_val = train_test_split(self.x_train, self.y_train, random_state=10, train_size=0.8, test_size=0.2)

        best_classifier = MLPClassifier(random_state= 10,
                                        solver = best_params_["solver"],
                                        hidden_layer_sizes = best_params_["hidden_layer_sizes"],
                                        activation = best_params_["activation"],
                                        learning_rate_init = best_params_["learning_rate_init"],
                                        max_iter = 500)
        
        for _ in range(epochs):
            best_classifier.partial_fit(x_train_split, y_train_split, classes=np.unique(y_train_split))
            scores_train.append(best_classifier.score(x_train_split, y_train_split))
            scores_val.append(best_classifier.score(x_val, y_val))
            
        debug(("scores_train_epoch: {0} \n").format(scores_train))
        debug(("scores_val_epoch: {0} \n").format(scores_val))
        debug(("loss: {0} \n").format(best_classifier.loss_curve_))

        plot_epochs(scores_train, scores_val, best_classifier.loss_curve_, self.folder+"neural_net_epochs", self.dataset[self.dataset_idx])
        
