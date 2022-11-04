from kmeans import *
from EM import *
from pca import *
from ica import *
from grp import *
from rf import *

def run_kmeans():
    km = K_Means(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], transformed_cols[dataset_idx], verbose = 0)
    km.run_kmeans()
    
    #best = 1
    #kmeans.best_cluster(best)

def run_EM():
    em = EM(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], transformed_cols[dataset_idx], verbose = 0)
    em.run_EM()
    

def run_pca():
    pca = PrincipalComponenetAnalysis(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "original/")
    pca.run_pca()
    
def run_ica():
    ica = IndependentComponentAnalysis(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "original/")
    ica.run_ica()
    
def run_grp():
    grp = RandomProjection(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "original/")
    grp.run_grp()
  
def run_rf():
    rf = RandomForest(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "original/")
    rf.run_rf()
      
def small_datatset():
    global x_train, y_train, x_test, y_test
    x_train, y_train, x_test, y_test = x_train.head(50), y_train.head(50), x_test.head(10), y_test.head(10)

if __name__ == "__main__":
    dataset_idx = 1
    if len(sys.argv) == 2: 
        dataset_idx = sys.argv[1]
        
    x_train, y_train, x_test, y_test = load_dataset_0() if dataset_idx == 0 else load_dataset_1()  
    datasets= ["bc", "titanic"]
    classes = [["Malignant", "Benign"], ["Not Survived","Survived"]]
    transformed_cols = [['mean radius', 'mean perimeter'], ['Age', 'Fare']]
    model_names = ["./models/dboost_bc.pkl", "./models/boost_titanic.pkl"]
    dataset = datasets[dataset_idx]
    #small_datatset()
    # Data Information
    print(("XTrain: {0} \n").format(x_train.describe()))
    print(("YTrain: {0} \n").format(y_train.info()))
    print(("YTrain Info: {0} \n").format(y_train.value_counts()))
    
    '''Experiments runs'''
    #run_kmeans()
    #run_EM()
    #run_pca()
    #run_ica()
    #run_grp()
    run_rf()