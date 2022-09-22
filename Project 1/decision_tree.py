import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_breast_cancer
import _pickle as cPickle
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def scale_data(x_train, x_test):
    scaler = StandardScaler()
    scaler = scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=x_train.columns)
    x_test_scaled = scaler.transform (x_test)
    x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=x_test.columns)
    return x_train_scaled_df, x_test_scaled_df
 
def resample_data(x_train, y_train):
    training  = pd.DataFrame()
    training[x_train.columns]=x_train
    training['target']=list(y_train)
    
    minority = training[training.target==1]
    majority = training[training.target==0]
    
    minority_upsampled = resample(minority,
                            replace=True,
                            n_samples=len(majority),
                            random_state=23)
    
    upsampled= pd.concat([majority,minority_upsampled])
    upsampled.target.value_counts()
    y_train = upsampled.target
    x_train = upsampled.drop('target', axis=1)
    return x_train, y_train
  
def feature_selection(x_train, x_test):
    corr_matrix = x_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    print("to drop", to_drop)
    x_train.drop(columns=to_drop, inplace=True)
    x_test.drop(columns=to_drop, inplace=True)
    return x_train, x_test

def load_dataset_1():
    # min = malignant
    data = load_breast_cancer(as_frame=True)
    data_df = data.frame
    x = data_df.drop(columns='target')
    #x['diagnosis'] = x['diagnosis'].apply(lambda x: '1' if x == 'M' else '0')
    y = data_df['target'].copy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    
    x_train, x_test = scale_data(x_train, x_test)
    x_train, x_test = feature_selection(x_train, x_test)
    
    x_train, y_train = resample_data(x_train, y_train)
    
    return x_train, y_train, x_test, y_test
 
 
def load_dataset_2():
    train_set = pd.read_csv('Data/titanic/train.csv')
    test_set = pd.read_csv('Data/titanic/test.csv')
    train_set = train_set.rename(columns= {"Survived" : "target"})
    y_train = train_set["target"].copy()

    x_train = train_set.drop(['target', 'Name', 'Embarked', 'Ticket', 'Cabin'], axis=1)
    x_train.replace({'male':0, 'female':1}, inplace = True )
    x_train.Age = x_train.Age.fillna(x_train.Age.mean())
    x_train = x_train
    
    x_test = test_set.drop(['Name', 'Embarked', 'Ticket', 'Cabin'], axis=1)
    x_test.replace({'male':0, 'female':1}, inplace = True )
    x_test.Age = x_test.Age.fillna(x_test.Age.mean())
    x_test.Fare = x_test.Fare.fillna(x_test.Fare.mean())

    submission_data = pd.read_csv('Data/titanic/gender_submission.csv')
    submission_data = submission_data.rename(columns= {"Survived" : "target"})
    y_test = submission_data['target'].copy()
    
    x_train, x_test = scale_data(x_train, x_test)
    x_train, y_train = resample_data(x_train, y_train)

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
    print(x_train.describe())
    print(y_train.info())
    print(y_train.value_counts())
    #train("decisiontree.pkl")