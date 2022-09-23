import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_breast_cancer
import _pickle as cPickle
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import time
from logging import debug
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.pyplot as pyplot
from sklearn.model_selection import GridSearchCV

def test(name, x_train, y_train, x_test, y_test, classes, dataset, filename):
    final_model = load_model(name)
    y_predict = final_model.predict(x_test)
    debug(final_model.score(x_train, y_train))
    debug(y_predict.shape)
    debug(y_test.shape)
    accuracy = accuracy_score(y_test, y_predict)
    debug(accuracy)
    metrics(y_test, y_predict, dataset, filename, classes)
    
def scale_data(x_train, x_test):
    scaler = StandardScaler()
    scaler = scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=x_train.columns)
    x_test_scaled = scaler.transform (x_test)
    x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=x_test.columns)
    return x_train_scaled_df, x_test_scaled_df
 
def resample_data(x_train, y_train):
    "ref: https://melaniesoek0120.medium.com/breast-cancer-classification-machine-learning-1150498f18e2"
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
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    debug("to drop", to_drop)
    x_train.drop(columns=to_drop, inplace=True)
    x_test.drop(columns=to_drop, inplace=True)
    return x_train, x_test

def load_dataset_0():
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
 
 
def load_dataset_1():
    train_set = pd.read_csv('./Data/titanic/train.csv')
    test_set = pd.read_csv('./Data/titanic/test.csv')
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

    submission_data = pd.read_csv('./Data/titanic/gender_submission.csv')
    submission_data = submission_data.rename(columns= {"Survived" : "target"})
    y_test = submission_data['target'].copy()
    
    x_train, x_test = scale_data(x_train, x_test)
    x_train, y_train = resample_data(x_train, y_train)
    return x_train, y_train, x_test, y_test

  
def save_model(final_model, name):
    with open(name, 'wb') as fid:
        cPickle.dump(final_model, fid)

def load_model(name):
    with open(name, 'rb') as fid:
        return cPickle.load(fid)

   
def cv_validation(classifier, param_name, param_range, x_train, y_train):
    train_score, test_score = validation_curve(classifier, x_train, y_train,
                                       param_name = param_name,
                                       param_range = param_range,
                                        cv = 5, scoring = "accuracy")
    return train_score, test_score
  
def learning(classifier, train_sizes, x_train, y_train):
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator = classifier,
        X = x_train, y = y_train, train_sizes = train_sizes, cv = 5, 
        shuffle = True, random_state = 45)
    return train_scores, validation_scores, train_sizes
 
 
def plot_train_val_curve(train_score, test_score, param_range, param_name, file_name, dataset):
    "ref: https://www.geeksforgeeks.org/validation-curve/"
    mean_train_score = np.mean(train_score, axis = 1)
    std_train_score = np.std(train_score, axis = 1)
    
    mean_test_score = np.mean(test_score, axis = 1)
    std_test_score = np.std(test_score, axis = 1)
    debug(param_name, "Mean Train Score", mean_train_score, "\n")
    debug(param_name, "Mean Test Score", mean_test_score, "\n")

    plt.plot(param_range, mean_train_score,
        label = "Training Score", color = 'b')
    plt.fill_between(
    param_range,
    mean_train_score - std_train_score,
    mean_train_score + std_train_score,
    alpha=0.2,
    color="b",
    lw=2,)
    
    plt.plot(param_range, mean_test_score,
        label = "Cross Validation Score", color = 'g')
    plt.fill_between(
    param_range,
    mean_test_score - std_test_score,
    mean_test_score + std_test_score,
    alpha=0.2,
    color="g",
    lw=2,)
    
    plt.title("Validation Curve with Decision Tree Classifier")
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc = 'best')
    plt.savefig("./images/"+dataset+file_name)
    plt.clf()
  
def plot_epochs(scores_train, scores_val, loss_curve, file_name, dataset):
    plt.plot(scores_train,
        label = "Training Score", color = 'b')
    
    plt.plot(scores_val,
        label = "Validation Score", color = 'g')
    
    plt.title("Accuracy over Epochs")
    plt.xlabel("# of Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc = 'best')
    plt.savefig("./images/"+dataset+file_name)
    plt.clf()
    
    plt.plot(loss_curve,
        label = "Loss", color = 'b')
    plt.title("Loss over Epochs")
    plt.xlabel("# of Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend(loc = 'best')
    plt.savefig("./images/"+dataset+file_name+"loss")
    plt.clf()

def get_train_time(best_classifier, x_train, y_train):
    start_time = time.time()
    best_classifier.fit(x_train, y_train)
    end_time = time.time()
    time_to_train = end_time - start_time
    debug("time to train", time_to_train)

def metrics(y_test, y_pred, dataset, filename, classes):
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    
    f1 = f1_score(y_test,y_pred)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    c_matrix = confusion_matrix(y_test,y_pred)
    debug("f1", f1)
    debug("accuracy", accuracy)
    debug("precision", precision)
    debug("recall", recall)
    debug("c_matrix", c_matrix, "\n")
    disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix,
                                  display_labels=["0", "1"])
    disp.plot()
    plt.savefig("./images/"+dataset+filename+"_cm")
    
    
def plot_compare_models(train_times, test_times, test_accuracies, classifiers):
  def times():
    fig, ax = pyplot.subplots()
    width = 0.3
    train_bar = ax.bar(xticks - width/2, train_times, width, label = 'Time to Train')
    test_bar = ax.bar(xticks + width/2, test_times, width, label = 'Time to Test')

    ax.set_ylabel('Time')
    ax.set_title('Time to Train and Test')
    ax.set_xticks(xticks)
    ax.set_xticklabels(classifiers)
    ax.bar_label(train_bar, padding=2)
    ax.bar_label(test_bar, padding=2)
    
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.clf()
  
  def scores():
    pyplot.bar(classifiers, test_accuracies)
    #pyplot.set_xticks(xticks)
    pyplot.set_xticklabels(classifiers)
    pyplot.ylabel("Test Accuracy")
    pyplot.title('Test Accuracy for the classifiers')
    plt.show()
    plt.clf()

  xticks = np.arange(len(classifiers))
  times()
  scores()


    
