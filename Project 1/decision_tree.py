from base import *
from sklearn.tree import DecisionTreeClassifier

def train_DT(model_name):
    
    # Post - pruning alpha
    classifier = DecisionTreeClassifier(random_state= 10)
    path = classifier.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas, _ = path.ccp_alphas, path.impurities
    debug(("CCP Alphas: {0} \n").format(ccp_alphas))
    
    parameters = {
        "criterion": ["entropy", "gini", "log_loss"],
        "min_samples_leaf": range(1, 10),
        "max_depth": range(1, 21),
        "ccp_alpha": ccp_alphas,
    }

    train_sizes =  np.linspace(0.1,1.0,10)
    
    train_score, test_score = cv_validation(classifier, "max_depth", parameters["max_depth"], x_train, y_train)
    plot_train_val_curve(train_score, test_score, parameters["max_depth"], "Max Depth","max_depth", dataset[dataset_idx])

    train_score, test_score = cv_validation(classifier, "ccp_alpha", parameters["ccp_alpha"], x_train, y_train)
    plot_train_val_curve(train_score, test_score, parameters["ccp_alpha"], "Cost of Pruning", "ccpa", dataset[dataset_idx])

    grid = GridSearchCV(classifier,
                        param_grid = parameters,
                        cv = 5,
                        verbose = True)
    grid.fit(x_train, y_train)
    
    final_model = grid.best_estimator_
    save_model(final_model, model_name)
    debug(("Best Parameters: {0} \n").format(grid.best_params_))
    
    best_classifier = DecisionTreeClassifier(random_state= 10,
                                             criterion = grid.best_params_["criterion"],
                                             min_samples_leaf = grid.best_params_["min_samples_leaf"],
                                             max_depth = grid.best_params_["max_depth"],
                                             ccp_alpha = grid.best_params_["ccp_alpha"])
    train_score, test_score, _ = learning(best_classifier, train_sizes, x_train, y_train)
    plot_train_val_curve(train_score, test_score, train_sizes, "Ratio of Train Sizes", "train_size", dataset[dataset_idx])

    get_train_time(best_classifier, x_train, y_train)
        
def test_DT(model_name):
    start_time = time.time()
    test(model_name, x_train, y_train, x_test, y_test, classes[dataset_idx], dataset[dataset_idx], "dt")
    end_time = time.time()
    time_to_test = end_time - start_time
    debug(("Time to Test: {0} \n").format(time_to_test))
    
if __name__ == "__main__":
    dataset_idx = 0
    if len(sys.argv) == 2: 
        dataset_idx = sys.argv[1]
        
    x_train, y_train, x_test, y_test = load_dataset_0() if dataset_idx == 0 else load_dataset_1()  
    
    dataset = ["bc_", "titanic_"]
    classes = [["Malignant", "Benign"], ["Not Survived","Survived"]]
    model_names = ["./models/decisiontree_bc.pkl", "./models/decisiontree_titanic.pkl"]
    
    # Data Information
    debug(("XTrain: \n {0} \n").format(x_train.describe()))
    debug(("XTrain: \n {0} \n").format(x_train.info()))
    debug(("YTrain: \n {0} \n").format(y_train.describe()))
    debug(("YTrain: \n {0} \n").format(y_train.info()))
    debug(("YTrain: \n {0} \n").format(y_train.value_counts()))
    
    train_DT(model_names[dataset_idx])
    test_DT(model_names[dataset_idx])