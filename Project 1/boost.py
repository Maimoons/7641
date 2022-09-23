from base import *
from sklearn.ensemble import GradientBoostingClassifier


def train_Boost(model_name):
        
    classifier = GradientBoostingClassifier(random_state= 10)
    
    parameters = {
        "n_estimators":  [10, 50, 100, 500, 1000, 5000],
        "learning_rate": np.logspace(-4,1,10),
    }

    train_sizes =  np.linspace(0.1,1.0,10)
    
    train_score, test_score = cv_validation(classifier, "n_estimators", parameters["n_estimators"], x_train, y_train)
    plot_train_val_curve(train_score, test_score, parameters["n_estimators"], "Number of Estimators","n_estimators", dataset[dataset_idx])

    train_score, test_score = cv_validation(classifier, "learning_rate", parameters["learning_rate"], x_train, y_train)
    plot_train_val_curve(train_score, test_score, parameters["learning_rate"], "Learning Rate", "learning_rate", dataset[dataset_idx])

    grid = GridSearchCV(classifier,
                        param_grid = parameters,
                        cv = 5,
                        verbose = True)
    grid.fit(x_train, y_train)
    
    final_model = grid.best_estimator_
    save_model(final_model, model_name)
    debug(("Best Parameters: {0} \n").format(grid.best_params_))
    
    best_classifier = GradientBoostingClassifier(random_state= 10,
                                             n_estimators = grid.best_params_["n_estimators"],
                                             learning_rate = grid.best_params_["learning_rate"])
    train_score, test_score, _ = learning(best_classifier, train_sizes, x_train, y_train)
    plot_train_val_curve(train_score, test_score, train_sizes, "Ratio of Train Sizes", "train_size", dataset[dataset_idx])

    get_train_time(best_classifier, x_train, y_train)
        
 
def test_Boost(model_name):
    start_time = time.time()
    test(model_name, x_train, y_train, x_test, y_test, classes[dataset_idx], dataset[dataset_idx], "boost")
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
    model_names = ["./models/dboost_bc.pkl", "./models/boost_titanic.pkl"]
    
    # Data Information
    debug(("XTrain: {0} \n").format(x_train.describe()))
    debug(("YTrain: {0} \n").format(y_train.info()))
    debug(("YTrain Info: {0} \n").format(y_train.value_counts()))
    
    train_Boost(model_names[dataset_idx])
    test_Boost(model_names[dataset_idx])

    