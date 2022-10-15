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
    plt.title("Validation Curve")
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc = 'best')
    plt.savefig("./images/"+dataset+file_name)
    plt.clf()
  
def plot_fitness_loss(classifier, file_name, dataset, best_params):
    fitness = classifier.fitness_curve
    
    plt.title("Fitness Curve \n"+ str(best_params))
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.plot(fitness[:,0])
    plt.savefig("./images/"+dataset+file_name)
    plt.clf()

def plot_all_fitness_loss(fitness_curve_list, algorithms, problem, size): 
    plt.title("Fitness Curve \n"+ problem + "\n Vector Size"+ str(size))
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    for idx, fitness in enumerate(fitness_curve_list):
        y = np.arange(1,fitness.shape[0]+1) 
        plt.plot(y, fitness, label = algorithms[idx])
    
    plt.legend(loc = 'best')
    plt.savefig("./images/"+problem+"/all_fitness"+str(size)+".png")
    plt.clf()
    
def plot_all_best_fitness(all_best_fitness_list, vectors, algorithms, problem):    
    plt.title("Best Fitness Curve \n"+ problem)
    plt.xlabel("Vector Length")
    plt.ylabel("Fitness")
    for idx, fitness in enumerate(all_best_fitness_list):
        plt.plot(vectors, fitness, label = algorithms[idx])
    plt.legend(loc = 'best')
    plt.savefig("./images/"+problem+"/all_best_fitness.png")
    plt.clf()
    
def plot_all_time_elapsed(all_time_elapsed_list, vectors, algorithms, problem):    
    plt.title("Time Elapsed \n"+ problem)
    plt.xlabel("Vector Length")
    plt.ylabel("Time")
    for idx, fitness in enumerate(all_time_elapsed_list):
        plt.plot(vectors, fitness, label = algorithms[idx])
    plt.legend(loc = 'best')
    plt.savefig("./images/"+problem+"/all_time.png")
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

    ax.legend()
    fig.tight_layout()
    plt.savefig("./images/"+dataset+"_models_time")
    plt.clf()
  
  def test_score():
    fig, ax = pyplot.subplots()
    bar = pyplot.bar(classifiers, test_accuracies)
    ax.bar_label(bar, padding=2)
    ax.set_ylabel("Test Accuracy")
    ax.set_title('Test Accuracy for the classifiers')
    #plt.xticks(rotation=90)
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
    #plt.xticks(rotation=90)
    ax.legend()
    fig.tight_layout()
    plt.savefig("./images/"+dataset+"_models_scores")
    plt.clf()

  xticks = np.arange(len(classifiers))
  times()
  test_score()
  scores()


classifiers = ["Random \n Hill", "Simulated \n Annealing", "Genetic \n Algorithm"]

titanic_train_time = [8.051584005355835, 2.6237659454345703, 59.07954668998718 ]
titanic_test_time = [ 0.13482427597045898 , 0.13295412063598633  , 0.09161114692687988]
titanic_test_score = [ 0.7105263157894737, 0.8971291866028708  , 0.8636363636363636]
titanic_f1 = [  0.5953177257525083 , 0.8739002932551321, 0.8403361344537815 ]
titanic_precision = [0.6054421768707483 , 0.7883597883597884, 0.7317073170731707]
titanic_recall = [0.5855263157894737, 0.9802631578947368, 0.9868421052631579]

plot_compare_models(titanic_train_time, titanic_test_time, titanic_test_score, titanic_f1, titanic_precision, titanic_recall, classifiers, "titanic")
 


