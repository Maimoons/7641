import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_breast_cancer
import _pickle as cPickle

def load_dataset_1():
    x = x.dropna()
    x_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
 
def load_dataset_2():
    train_set = pd.read_csv('Data/titanic/train.csv')
    test_set = pd.read_csv('Data/titanic/test.csv')

    y_train = train_set['Survived'].copy().head(10)

    x_train = train_set.drop(['Survived', 'Name', 'Embarked', 'Ticket', 'Cabin'], axis=1)
    x_train.replace({'male':0, 'female':1}, inplace = True )
    x_train.Age = x_train.Age.fillna(x_train.Age.mean())
    x_train = x_train.head(10)
    
    x_test = test_set.drop(['Name', 'Embarked', 'Ticket', 'Cabin'], axis=1)
    x_test.replace({'male':0, 'female':1}, inplace = True )
    x_test.Age = x_test.Age.fillna(x_test.Age.mean())
    x_test.Fare = x_test.Fare.fillna(x_test.Fare.mean())

    submission_data = pd.read_csv('Data/titanic/gender_submission.csv')
    y_test = submission_data['Survived'].copy()
    
    print(train_set.columns)
    print(x_train.head())
    print(x_train.isnull().sum())

    print(test_set.columns)
    print(x_test.head())
    print(x_test.isnull().sum())
    return x_train, y_train, x_test, y_test

def train(model_name):
    
    # Post - pruning alpha
    classifier = DecisionTreeClassifier(random_state= 1024)
    path = classifier.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    parameters = {
        "criterion": ["entropy", "gini", "log_loss"],
        "min_samples_leaf": range(1, 10),
        "max_depth": range(1, 5),
        "ccp_alpha": ccp_alphas
    }

    """ classifiers = []

    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(x_train, y_train)
        classifiers.append(clf)
        
        
    acc_scores = [accuracy_score(y_test, classifier.predict(X_test)) for classifier in classifiers]
    print(acc_scores)
    """

    grid = GridSearchCV(classifier,
                        param_grid = parameters,
                        cv = 5,
                        verbose = True)
    grid.fit(x_train, y_train)
    final_model = grid.best_estimator_
    save_model(final_model, model_name)

    test(model_name, x_train, y_train, x_test)

def test(name, x_train, y_train, x_test):
    final_model = load_model(name)
    y_predict = final_model.predict(x_test)
    print(final_model.score(x_train, y_train))
    print(y_predict.shape)
    print(y_test.shape)
    accuracy = accuracy_score(y_test, y_predict)
    print(accuracy)
    
def save_model(final_model, name):
    with open(name, 'wb') as fid:
        cPickle.dump(final_model, fid)

def load_model(name):
    with open(name, 'rb') as fid:
        return cPickle.load(fid)
    
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_dataset_2()
    train("decisiontree.pkl")