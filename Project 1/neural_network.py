from base import *
from sklearn.neural_network import MLPClassifier


def train_ANN(model_name):    
    classifier = MLPClassifier(random_state= 10, max_iter = 1000)
    
    parameters = {
        "solver": ['lbfgs', 'adam'],
        "hidden_layer_sizes": np.arange(10, 100, step=10),
        "activation": ['tanh', 'relu'],
        "learning_rate_init": np.logspace(-4,1,10),
    }
    train_sizes =  np.linspace(0.1,1.0,10)
    
    train_score, test_score = cv_validation(classifier, "hidden_layer_sizes", parameters["hidden_layer_sizes"], x_train, y_train)
    plot_train_val_curve(train_score, test_score, parameters["hidden_layer_sizes"], "Hidden units","hidden_layer_sizes", dataset[dataset_idx])

    train_score, test_score = cv_validation(classifier, "learning_rate_init", parameters["learning_rate_init"], x_train, y_train)
    plot_train_val_curve(train_score, test_score, parameters["learning_rate_init"], "Learning Rate", "learning_rate_init", dataset[dataset_idx])

    grid = GridSearchCV(classifier,
                        param_grid = parameters,
                        cv = 5,
                        verbose = True)
    grid.fit(x_train, y_train)
    
    final_model = grid.best_estimator_
    save_model(final_model, model_name)
    debug("best parameters", grid.best_params_)
    
    best_classifier = MLPClassifier(random_state= 10,
                                    solver = grid.best_params_["solver"],
                                    hidden_layer_sizes = grid.best_params_["hidden_layer_sizes"],
                                    activation = grid.best_params_["activation"],
                                    learning_rate_init = grid.best_params_["learning_rate_init"],
                                    max_iter= 500)
    
    train_score, test_score, _ = learning(best_classifier, train_sizes, x_train, y_train)
    plot_train_val_curve(train_score, test_score, train_sizes, "Ratio of Train Sizes", "train_size", dataset[dataset_idx])

    run_epochs(grid.best_params_, 300)
    get_train_time(best_classifier, x_train, y_train)
     
def test_ANN(model_name):
    start_time = time.time()
    test(model_name, x_train, y_train, x_test, y_test, classes[dataset_idx], dataset[dataset_idx], "nn")
    end_time = time.time()
    time_to_test = end_time - start_time
    debug("time to test", time_to_test) 
                 
def run_epochs(best_params_, epochs):
    scores_train = []
    scores_val = []
    
    x_train_split, x_val, y_train_split, y_val = train_test_split(x_train, y_train, random_state=10, train_size=0.8, test_size=0.2)

    best_classifier = MLPClassifier(random_state= 10,
                                    solver = best_params_["solver"],
                                    hidden_layer_sizes = best_params_["hidden_layer_sizes"],
                                    activation = best_params_["activation"],
                                    learning_rate_init = best_params_["learning_rate_init"],
                                    max_iter = 500)
    
    for epoch in range(epochs):
        best_classifier.partial_fit(x_train_split, y_train_split, classes=np.unique(y_train_split))
        scores_train.append(best_classifier.score(x_train_split, y_train_split))
        scores_val.append(best_classifier.score(x_val, y_val))
        
    debug("scores_train_epoch", scores_train)
    debug("scores_val_epoch", scores_val)
    debug("loss", best_classifier.loss_curve_)
    plot_epochs(scores_train, scores_val, best_classifier.loss_curve_, "neural_net_epochs", dataset)
  
def run_epochs_classifier(best_classifier, epochs):
    scores_train = []
    scores_val = []
    
    x_train_split, x_val, y_train_split, y_val = train_test_split(x_train, y_train, random_state=10, train_size=0.8, test_size=0.2)
    
    for epoch in range(epochs):
        best_classifier.partial_fit(x_train_split, y_train_split)
        scores_train.append(best_classifier.score(x_train_split, y_train_split))
        scores_val.append(best_classifier.score(x_val, y_val))
        
    debug("scores_train_epoch", scores_train)
    debug("scores_val_epoch", scores_val)
    debug("loss", best_classifier.loss_curve_)
    plot_epochs(scores_train, scores_val, best_classifier.loss_curve_, "neural_net_epochs", dataset[dataset_idx])
      
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_dataset_1()
    
    dataset_idx = 1
    dataset = ["bc_", "titanic_"]
    classes = [["Malignant", "Benign"], ["Not Survived","Survived"]]
    model_names = ["./models/decisiontree_bc.pkl", "./models/decisiontree_titanic.pkl"]

    debug(x_train.describe())
    debug(y_train.info())
    debug(y_train.value_counts())
    debug("\n \n")
    train_ANN(model_names[dataset_idx])
    test_ANN(model_names[dataset_idx])
