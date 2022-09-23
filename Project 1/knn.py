from base import *
from sklearn.neighbors import KNeighborsClassifier


def train_KNN(model_name):
    
    classifier = KNeighborsClassifier(random_state= 10, algorithm='auto')
    
    parameters = {
        "n_neighbors":  np.arange(1,21,2),
        "weights": ["uniform", "distance"],
        "p": [1,2],
        "metric": ["minkowski", "chebyshev", "braycurtis", "canberra", "cityblock", "correlation", "cosine", "euclidean", "sqeuclidean"]
    }

    train_sizes =  np.linspace(0.1,1.0,10)
    
    train_score, test_score = cv_validation(classifier, "n_neighbors", parameters["n_neighbors"], x_train, y_train)
    plot_train_val_curve(train_score, test_score, parameters["n_neighbors"], "Number of Neighbors","n_neighbors", dataset[dataset_idx])

    train_score, test_score = cv_validation(classifier, "metric", parameters["metric"], x_train, y_train)
    plot_train_val_curve(train_score, test_score, parameters["metric"], "Distance Metric", "metric", dataset[dataset_idx])

    grid = GridSearchCV(classifier,
                        param_grid = parameters,
                        cv = 5,
                        verbose = True)
    grid.fit(x_train, y_train)
    
    final_model = grid.best_estimator_
    save_model(final_model, model_name)
    debug(("Best Parameters: {0} \n").format(grid.best_params_))
    
    best_classifier = KNeighborsClassifier(random_state= 10,
                                             algorithm='auto',
                                             n_neighbors = grid.best_params_["n_neighbors"],
                                             weights = grid.best_params_["weights"],
                                             p = grid.best_params_["p"],
                                             metric = grid.best_params_["metric"])
    train_score, test_score, _ = learning(best_classifier, train_sizes, x_train, y_train)
    plot_train_val_curve(train_score, test_score, train_sizes, "Ratio of Train Sizes", "train_size", dataset[dataset_idx])

    get_train_time(best_classifier, x_train, y_train)

   
def test_KNN(model_name):
    start_time = time.time()
    test(model_name, x_train, y_train, x_test, y_test, classes[dataset_idx])
    end_time = time.time()
    time_to_test = end_time - start_time
    debug(("Time to Test: {0} \n").format(time_to_test))
     
if __name__ == "__main__":
    dataset_idx = 1
    if len(sys.argv) != 2: 
        dataset_idx = sys.argv[1]
        
    x_train, y_train, x_test, y_test = load_dataset_0() if dataset_idx == 0 else load_dataset_1()  
    dataset = ["bc_", "titanic_"]
    classes = [["Malignant", "Benign"], ["Not Survived","Survived"]]
    model_names = ["./models/knn_bc.pkl", "./models/knn_titanic.pkl"]
    
    # Data Information
    debug(("XTrain: {0} \n").format(x_train.describe()))
    debug(("YTrain: {0} \n").format(y_train.info()))
    debug(("YTrain Info: {0} \n").format(y_train.value_counts()))
    
    train_KNN(model_names[dataset_idx])
    test_KNN(model_names[dataset_idx])
