from base import *
from sklearn import svm


def train_SVM(model_name):
    
    classifier = svm.SVC(random_state= 10)
    
    parameters = {
        "kernel":  ["linear", "poly", "rbf", "sigmoid"],
        "C": np.logspace(-4,1,10),
        "gamma": np.logspace(-4,10,10, base = 2),
    }

    train_sizes =  np.linspace(0.1,1.0,10)
    
    train_score, test_score = cv_validation(classifier, "C", parameters["C"], x_train, y_train)
    plot_train_val_curve(train_score, test_score, parameters["C"], "Regularization Parameter","C", dataset[dataset_idx])

    train_score, test_score = cv_validation(classifier, "gamma", parameters["gamma"], x_train, y_train)
    plot_train_val_curve(train_score, test_score, parameters["gamma"], "Kernel Coefficient", "gamma", dataset[dataset_idx])

    grid = GridSearchCV(classifier,
                        param_grid = parameters,
                        cv = 5,
                        verbose = True)
    grid.fit(x_train, y_train)
    
    final_model = grid.best_estimator_
    save_model(final_model, model_name)
    debug("best parameters", grid.best_params_)
    
    best_classifier = svm.SVC(random_state= 10,
                                             kernel = grid.best_params_["kernel"],
                                             C = grid.best_params_["C"],
                                             gamma = grid.best_params_["gamma"])
    train_score, test_score, _ = learning(best_classifier, train_sizes, x_train, y_train)
    plot_train_val_curve(train_score, test_score, train_sizes, "Ratio of Train Sizes", "train_size", dataset[dataset_idx])

    get_train_time(best_classifier, x_train, y_train)
        
 
def test_SVM(model_name):   
    start_time = time.time()
    test(model_name, x_train, y_train, x_test, y_test, classes[dataset_idx])
    end_time = time.time()
    time_to_test = end_time - start_time
    debug("time to test", time_to_test)
    
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
    
    train_SVM(model_names[dataset_idx])
    test_SVM(model_names[dataset_idx])
