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

def test(name, x_train, y_train, x_test, y_test, classes, dataset, filename, verbose):
    final_model = load_model(name)
    y_predict = final_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    print(("Test Accuracy: {0} \n").format(accuracy))
    print(("Final Train Accuracy: {0} \n").format(final_model.score(x_train, y_train)))

    metrics(y_test, y_predict, dataset, filename, verbose=verbose)
    
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
    #x_train, x_test = feature_selection(x_train, x_test)
    #x_train, y_train = resample_data(x_train, y_train)
    
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

def metrics(y_test, y_pred, dataset, filename, verbose):
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
    
    if verbose:
        print(("f1: {0} \n accuracy: {1} precision: {2} \n recall: {3}").format(f1, accuracy, precision, recall))
        #print(("Confusion Matrix: {0} \n").format(c_matrix))
        disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix,
                                  display_labels=["0", "1"])
        disp.plot()
        plt.savefig("./images/"+dataset+filename+"_cm")
        plt.clf()
        
    return f1, accuracy, precision, recall
    
    
def plot_compare_models(train_times, test_accuracies, f1, precision, recall, classifiers, dataset):
  def times():
    fig, ax = pyplot.subplots()
    width = 0.3
    train_bar = ax.bar(xticks - width/2, train_times, width, label = 'Time to Train')

    ax.set_ylabel('Time')
    ax.set_title('Time to Train')
    ax.set_xticks(xticks)
    ax.set_xticklabels(classifiers)
    ax.bar_label(train_bar, padding=2)

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

def plot_compare_kmeans(wcss, distortion, hmg, sil, test_accuracies, f1, precision, recall, classifiers, dataset):
    def plot_wcss():
        fig, ax = pyplot.subplots()
        width = 0.4
        wcss_bar = ax.bar(xticks+0.8, wcss, width, label = 'Mean Square Error')

        ax.set_ylabel('Scores')
        ax.set_title('KMeans with dimenaionsinality reduction - Mean Square Error')
        ax.set_xticks(xticks+2*width)
        ax.set_xticklabels(classifiers)
        #plt.xticks(rotation=90)
        ax.legend()
        fig.tight_layout()
        plt.savefig("./images/"+dataset+"_models_wcss")
        plt.clf()
        
    def test_score():
        fig, ax = pyplot.subplots()
        width = 0.2
        distortion_bar = ax.bar(xticks, distortion, width, label = 'Distortion')
        hmg_bar = ax.bar(xticks + width, hmg, width, label = 'Homogenity')
        sil_bar = ax.bar(xticks + 2*width, sil, width, label = 'Silhouette')

        ax.set_ylabel('Scores')
        ax.set_title('KMeans with dimenaionsinality reduction')
        ax.set_xticks(xticks+2*width)
        ax.set_xticklabels(classifiers)
        #plt.xticks(rotation=90)
        ax.legend()
        fig.tight_layout()
        plt.savefig("./images/"+dataset+"_km_models_scores")
        plt.clf()
    
    def scores():
        fig, ax = pyplot.subplots()
        width = 0.2
        accuracy_bar = ax.bar(xticks, test_accuracies, width, label = 'Accuracy')
        f1_bar = ax.bar(xticks + width, f1, width, label = 'F1')
        precision_bar = ax.bar(xticks + 2*width, precision, width, label = 'Precision')
        recall_bar = ax.bar(xticks + 3*width, recall, width, label = 'Recall')

        ax.set_ylabel('Scores')
        ax.set_title('KMeans with dimenaionsinality reduction')
        ax.set_xticks(xticks+2*width)
        ax.set_xticklabels(classifiers)
        #plt.xticks(rotation=90)
        ax.legend(loc='lower right')
        fig.tight_layout()
        plt.savefig("./images/"+dataset+"_models_scores")
        plt.clf()

    xticks = np.arange(len(classifiers))
    scores()
    test_score()
    plot_wcss()
    
def plot_compare_EM(score, hmg, sil, aic, bic, test_accuracies, f1, precision, recall, classifiers, dataset):
    def aic_bic():
        fig, ax = pyplot.subplots()
        width = 0.2
        aic_bar = ax.bar(xticks, aic, width, label = 'AIC')
        bic_bar = ax.bar(xticks + width, bic, width, label = 'BIC')

        ax.axhline(0, c="black", lw=1, ls="--")
        ax.set_ylabel('Scores')
        ax.set_title('EM with dimenaionsinality reduction - AIC and BIC')
        ax.set_xticks(xticks+3*width)
        ax.set_xticklabels(classifiers)
        #plt.xticks(rotation=90)
        ax.legend()
        fig.tight_layout()
        plt.savefig("./images/"+dataset+"_aic_bic")
        plt.clf()
        
    def loglikelihood():
        fig, ax = pyplot.subplots()
        width = 0.4
        score_bar = ax.bar(xticks+1.2, score, width, label = 'score')
        ax.axhline(0, c="black", lw=1, ls="--")
        ax.set_ylabel('Scores')
        ax.set_title('EM with dimenaionsinality reduction - Log likelihood')
        ax.set_xticks(xticks+3*width)
        ax.set_xticklabels(classifiers)
        #plt.xticks(rotation=90)
        ax.legend()
        fig.tight_layout()
        plt.savefig("./images/"+dataset+"_mle")
        plt.clf()
    
    def test_score():
        fig, ax = pyplot.subplots()
        width = 0.2
        hmg_bar = ax.bar(xticks + 2*width, hmg, width, label = 'Homogenity')
        sil_bar = ax.bar(xticks + 3*width, sil, width, label = 'Silhouette')

        ax.set_ylabel('Scores')
        ax.set_title('EM with dimenaionsinality reduction')
        ax.set_xticks(xticks+3*width)
        ax.set_xticklabels(classifiers)
        #plt.xticks(rotation=90)
        ax.legend()
        fig.tight_layout()
        plt.savefig("./images/"+dataset+"_em_models_scores")
        plt.clf()
    
    def scores():
        fig, ax = pyplot.subplots()
        width = 0.2
        accuracy_bar = ax.bar(xticks, test_accuracies, width, label = 'Accuracy')
        f1_bar = ax.bar(xticks + width, f1, width, label = 'F1')
        precision_bar = ax.bar(xticks + 2*width, precision, width, label = 'Precision')
        recall_bar = ax.bar(xticks + 3*width, recall, width, label = 'Recall')

        ax.set_ylabel('Scores')
        ax.set_title('KMeans with dimenaionsinality reduction')
        ax.set_xticks(xticks+2*width)
        ax.set_xticklabels(classifiers)
        #plt.xticks(rotation=90)
        ax.legend(loc='lower right')
        fig.tight_layout()
        plt.savefig("./images/"+dataset+"_models_scores")
        plt.clf()

    xticks = np.arange(len(classifiers))
    scores()
    test_score()
    loglikelihood()
    aic_bic()
  
classifiers = ["Original","PCA", "ICA", "Randdom", "Random Forest"]

titanic_f1 = [0.734375, 0.7259036144578314, 0.7267267267267269, 0.6481178396072014, 0.734375     ]
titanic_test_score = [0.8092031425364759, 0.7957351290684624, 0.795735129068462, 0.7586980920314254, 0.8092031425364759]
titanic_precision = [0.7885906040268457 , 0.7484472049689441, 0.7469135802469136, 0.7360594795539034, 0.7885906040268457    ]
titanic_recall = [0.6871345029239766, 0.7046783625730995, 0.7076023391812866, 0.5789473684210527, 0.6871345029239766]

wcss = [2689.517965780913  , 2387.8456389884273, 3262.073076349175, 1799.667256954667, 2689.517965780913   ]

distortion = [1.5218771331910759  , 1.4536358526220123, 1.730450108471777, 1.2097308514104386, 1.5218771331910759    ]
hmg = [0.31225574158660535  , 0.2955972991380535, 0.26730203957065585, 0.181341091341289 , 0.31225574158660535  ]
sil = [0.2450891339945855  , 0.26119513832634567, 0.2483420960176198, 0.34202242469515187, 0.2450891339945855    ]
 
#plot_compare_kmeans(wcss, distortion, hmg, sil,\
#    titanic_test_score, titanic_f1, titanic_precision, titanic_recall, classifiers, "titanic/kmeans/")

bc_f1 = [0.9227722772277228, 0.954627949183303, 0.8404423380726699, 0.9363957597173144, 0.9618874773139746      ]
bc_test_score = [0.9084507042253521, 0.9413145539906104, 0.7629107981220657, 0.9154929577464789, 0.9507042253521126]
bc_precision = [0.9789915966386554 , 0.926056338028169, 0.726775956284153, 0.8862876254180602, 0.9330985915492958   ]
bc_recall = [0.8726591760299626, 0.9850187265917603, 0.9962546816479401, 0.9925093632958801, 0.9925093632958801]

wcss = [5664.307347720415 ,6004.572857272221 ,  9343.989556771947,  5249.395611037546, 4617.807029244761     ]

distortion = [3.3504661020007287 , 3.4304347621796745, 4.294740847262001, 3.234729788792857, 3.022282396264803      ]
hmg = [0.7192734628983963 ,  0.6880160434561106, 0.2584167508967814, 0.5947542265328442, 0.7511686543128645   ]
sil = [0.1480628037076431 ,  0.17949187266548913, -0.0013827321855904907, 0.20266885630090437, 0.1773307512512986     ]
 
#plot_compare_kmeans(wcss, distortion, hmg, sil,\
#    bc_test_score, bc_f1, bc_precision, bc_recall, classifiers, "bc/kmeans/")



titanic_f1 = [ 0.6923076923076924 , 0.7067901234567903, 0.7220543806646524 , 0.7291361639824304, 0.6923076923076924    ]
titanic_test_score = [ 0.8024691358024691, 0.7867564534231201, 0.7934904601571269, 0.792368125701459, 0.8024691358024691  ]
titanic_precision = [0.8608695652173913,  0.7483660130718954 , 0.746875, 0.7302052785923754, 0.8608695652173913   ]
titanic_recall = [0.5789473684210527, 0.6695906432748538, 0.6988304093567251, 0.7280701754385965, 0.5789473684210527]
hmg = [ 0.33436583032295697 ,  0.3300723182738356 , 0.3500641649085938, 0.2914522194175769, 0.33436583032295697   ]
sil = [0.09926366500529733 , 0.09140915642999209, 0.1367123518569962, 0.018796650815006005 , 0.09926366500529733 ]

score = [11.830960049313694 , 11.449200629978147, 10.712976793192551, 2.411946265419303, 11.830960049313694    ]
aic = [-20694.770807877,  -19324.47552262106, -18012.524645469126, -3670.088244977198 , -20694.770807877    ]
bic = [-19765.055988947664, -16741.40187621429, -15429.45099906236, -2165.292094751364, -19765.055988947664    ]
 
#plot_compare_EM(score, hmg, sil, aic, bic,\
#    titanic_test_score, titanic_f1, titanic_precision, titanic_recall, classifiers, "titanic/EM/")

bc_f1 = [ 0.9467680608365018 ,0.7636363636363636, 0.885608856088561 , 0.951310861423221, 0.9225092250922509 ]
bc_test_score = [0.9342723004694836, 0.755868544600939, 0.8544600938967136, 0.9389671361502347, 0.9014084507042254]
bc_precision = [0.9613899613899614, 0.9710982658959537 , 0.8727272727272727,  0.951310861423221, 0.9090909090909091    ]
bc_recall = [0.9325842696629213, 0.6292134831460674, 0.898876404494382, 0.951310861423221, 0.9363295880149812]
hmg = [0.6803812050237493 , 0.3188536517998427, 0.44451817405963023, 0.6712238313015945, 0.5298990433315339     ]
sil = [0.2901814275824707 , 0.11912626798232562, 0.0038930587136564794, 0.3009469673307946, 0.32411435829939134    ]

score = [ 4.6299819995955875 ,   -2.632218679856009, -28.604631435367597, 19.635893309428457, 1.8583399916463714    ]
aic = [ -970.7446636554405 , 2606.65031523732, 25469.145982933194, -13755.781099633045, 364.69432711729155    ]
bic = [5058.206644247114 , 3344.5582762583454, 27695.033184035077, -7726.82979173049, 4313.718250383658    ]

#plot_compare_EM(score, hmg, sil, aic, bic,\
#    bc_test_score, bc_f1, bc_precision, bc_recall, classifiers, "bc/EM/")



#Neural Network Dim Red
bc_train_time = [0.8072903156280518  , 0.70107102394104, 0.6617217063903809, 0.5888528823852539 , 0.3736550807952881  ]
#bc_test_time = []
bc_f1 = [0.9230769230769231, 0.9726775956284154, 0.956043956043956, 0.9720670391061451, 0.9777777777777777    ]
bc_test_score = [0.9371428571428572, 0.965034965034965, 0.9440559440559441, 0.965034965034965, 0.972027972027972   ]
bc_precision = [0.9647058823529412 , 0.956989247311828 , 0.9456521739130435, 0.9775280898876404 , 0.9777777777777777   ]
bc_recall = [ 0.9111111111111111, 0.9888888888888889, 0.9666666666666667, 0.9666666666666667, 0.9777777777777777]

#plot_compare_models(bc_train_time, bc_test_score, bc_f1, bc_precision, bc_recall, classifiers, "bc/neural_network/")
 
titanic_train_time = [0.0007789134979248047 , 1.0145819187164307,  0.8888561725616455, 0.6556663513183594, 0.3938629627227783     ]
#titanic_test_time = [ ]
titanic_f1 = [0.8229665071770335, 0.8431372549019608 , 0.6591760299625469 , 0.7014925373134329, 0.6360424028268551   ]
titanic_test_score = [0.7861271676300577, 0.885167464114832, 0.7822966507177034, 0.8086124401913876, 0.7535885167464115]
titanic_precision = [ 0.7010309278350515, 0.8376623376623377, 0.7652173913043478 , 0.8103448275862069, 0.6870229007633588   ]
titanic_recall = [0.8947368421052632 , 0.8486842105263158, 0.5789473684210527, 0.618421052631579, 0.5921052631578947]

#plot_compare_models(titanic_train_time, titanic_test_score, titanic_f1, titanic_precision, titanic_recall, classifiers, "titanic/neural_network/")
 

#Neural Network Clustering

bc_train_time = [0.8072903156280518 , 0.36431002616882324, 0.677401065826416, 0.11473488807678223, 0.3572349548339844   ]
#bc_test_time = []
bc_f1 = [0.9230769230769231, 0.9723756906077348, 0.9565217391304347, 0.9608938547486034, 0.9666666666666667   ]
bc_test_score = [0.9371428571428572, 0.965034965034965, 0.9440559440559441, 0.951048951048951, 0.958041958041958 ]
bc_precision = [0.9647058823529412 , 0.967032967032967, 0.9361702127659575 , 0.9662921348314607, 0.9666666666666667   ]
bc_recall = [0.9111111111111111, 0.9777777777777777, 0.9777777777777777 , 0.9555555555555556, 0.9666666666666667]

plot_compare_models(bc_train_time, bc_test_score, bc_f1, bc_precision, bc_recall, classifiers, "bc/neural_network/clustering/")
 
titanic_train_time = [ 0.0007789134979248047 , 0.8562860488891602, 0.8787479400634766, 0.7603981494903564, 1.066239833831787   ]
#titanic_test_time = [ ]
titanic_f1 = [0.8229665071770335, 0.7220216606498195,  0.8078175895765473, 0.6825396825396826, 0.6484375   ]
titanic_test_score = [0.7861271676300577, 0.8157894736842105, 0.8588516746411483, 0.8086124401913876, 0.784688995215311]
titanic_precision = [ 0.7010309278350515, 0.8, 0.8 , 0.86 , 0.7980769230769231  ]
titanic_recall = [0.8947368421052632 , 0.6578947368421053, 0.8157894736842105, 0.5657894736842105, 0.5460526315789473 ]

plot_compare_models(titanic_train_time, titanic_test_score, titanic_f1, titanic_precision, titanic_recall, classifiers, "titanic/neural_network/clustering/")
 

