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
import sys  		  	   		  	  		  		  		    	 		 		   		 		  

def test(name, x_train, y_train, x_test, y_test, classes, dataset, filename):
    final_model = load_model(name)
    y_predict = final_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(("Test Accuracy: {0} \n").format(accuracy))
    print(("Final Train Accuracy: {0} \n").format(final_model.score(x_train, y_train)))

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
    x_train.drop(columns=to_drop, inplace=True)
    x_test.drop(columns=to_drop, inplace=True)
    print(("Columns to drop: {0} \n").format(to_drop))

    return x_train, x_test

def load_dataset_0():
    # min = malignant
    data = load_breast_cancer(as_frame=True)
    data_df = data.frame
    x = data_df.drop(columns='target')
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
    #x_train, y_train = resample_data(x_train, y_train)
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
        shuffle = True, random_state = 10)
    return train_scores, validation_scores, train_sizes
 
 
def plot_train_val_curve(train_score, test_score, param_range, param_name, file_name, dataset):
    "ref: https://www.geeksforgeeks.org/validation-curve/"
    mean_train_score = np.mean(train_score, axis = 1)
    std_train_score = np.std(train_score, axis = 1)
    
    mean_test_score = np.mean(test_score, axis = 1)
    std_test_score = np.std(test_score, axis = 1)
    print(param_name, "Mean Train Score", mean_train_score, "\n")
    print(param_name, "Mean Test Score", mean_test_score, "\n")

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
    plt.xticks(rotation=90)
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
    print(("Time to Train: {0} \n").format(time_to_train))

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
    
    print(("f1: {0} \n accuracy: {1} precision: {2} \n recall: {3}").format(f1, accuracy, precision, recall))
    print(("Confusion Matrix: {0} \n").format(c_matrix))

    disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix,
                                  display_labels=["0", "1"])
    disp.plot()
    plt.savefig("./images/"+dataset+filename+"_cm")
    
    
def plot_compare_models(train_times, test_times, test_accuracies, f1, precision, recall, classifiers, dataset):
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
    plt.xticks(rotation=90)

    ax.legend()
    fig.tight_layout()
    plt.savefig("./images/"+dataset+"_models_time")
    plt.clf()
  
  def test_score():
    fig, ax = pyplot.subplots()
    bar = pyplot.bar(classifiers, test_accuracies)
    #pyplot.set_xticks(xticks)
    #pyplot.set_xticklabels(classifiers)
    ax.bar_label(bar, padding=2)
    ax.set_ylabel("Test Accuracy")
    ax.set_title('Test Accuracy for the classifiers')
    plt.xticks(rotation=90)
    fig.tight_layout()
    plt.savefig("./images/"+dataset+"_models_test")
    plt.clf()
  
  def scores():
    fig, ax = pyplot.subplots()
    width = 0.2
    accuracy_bar = ax.bar(xticks, test_accuracies, width, label = 'Accuracy')
    f1_bar = ax.bar(xticks + width, f1, width, label = 'F1')
    precision_bar = ax.bar(xticks + 2*width, precision, width, label = 'Precision')
    recall_bar = ax.bar(xticks + 3*width, recall, width, label = 'Recall')

    ax.set_ylabel('Scores')
    ax.set_title('Test Metrics')
    ax.set_xticks(xticks+2*width)
    ax.set_xticklabels(classifiers)
    plt.xticks(rotation=90)
    ax.legend()
    fig.tight_layout()
    plt.savefig("./images/"+dataset+"_models_scores")
    plt.clf()

  xticks = np.arange(len(classifiers))
  times()
  test_score()
  scores()

classifiers = ["DT", "Boosted DT", "ANN", "SVM", "KNN"]

bc_train_time = [0.0018067359924316406, 0.06789684295654297, 0.14522290229797363, 0.001252889633178711, 0.0005869865417480469]
bc_test_time = [0.06498098373413086, 0.07267117500305176, 0.07519102096557617, 0.06551814079284668, 0.07495284080505371]
bc_test_score = [0.8881118881118881, 0.9370629370629371, 0.9905660377358491, 0.951048951048951, 0.9230769230769231]
bc_f1 = [0.9090909090909092, 0.9425287356321839, 0.96045197740113, 0.96 , 0.9371428571428572]
bc_precision = [0.9302325581395349, 0.9764705882352941, 0.9770114942528736, 0.9882352941176471, 0.9647058823529412 ]
bc_recall = [ 0.8888888888888888, 0.9222222222222223, 0.9444444444444444, 0.9333333333333333, 0.9111111111111111]

#plot_compare_models(bc_train_time, bc_test_time, bc_test_score, bc_f1, bc_precision, bc_recall, classifiers, "bc")
 
titanic_train_time = [0.0017900466918945312, 0.014275074005126953, 1.0585441589355469, 0.011755943298339844, 0.0007789134979248047  ]
titanic_test_time = [ 0.06725382804870605, 0.06751894950866699 , 0.09732794761657715, 0.1081380844116211, 0.08622407913208008]
titanic_test_score = [0.8881118881118881, 0.8301435406698564 , 0.84688995215311, 0.8492822966507177,  0.8229665071770335]
titanic_f1 = [ 0.9090909090909092 , 0.8135593220338982, 0.8048780487804879, 0.819484240687679, 0.7861271676300577]
titanic_precision = [0.9302325581395349, 0.6956521739130435, 0.75, 0.7258883248730964, 0.7010309278350515 ]
titanic_recall = [0.8888888888888888, 0.9473684210526315, 0.868421052631579, 0.9407894736842105, 0.8947368421052632 ]

#plot_compare_models(titanic_train_time, titanic_test_time, titanic_test_score, titanic_f1, titanic_precision, titanic_recall, classifiers, "titanic")
 
