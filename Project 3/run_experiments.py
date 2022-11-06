from kmeans import *
from EM import *
from pca import *
from ica import *
from grp import *
from rf import *
from neural_network import *

def run_kmeans():
    km = K_Means(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], transformed_cols[dataset_idx], verbose = 0, run_plot=True)
    km.run_kmeans()
 
    
def run_kmeans_with_dim_reduction():
    #pca
    pca = PrincipalComponenetAnalysis(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "original/", run_plot=False)
    x_pca, x_test_pca = pca.run()
    km = K_Means(x_pca, y_train, x_test_pca, y_test, dataset, classes[dataset_idx], transformed_cols[dataset_idx], verbose = 0, folder = "pca/", run_plot=True)
    km.run_kmeans()
    
    #ica
    ica = IndependentComponentAnalysis(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "original/", run_plot=False)
    x_ica, x_test_ica = ica.run()
    km = K_Means(x_ica, y_train, x_test_ica, y_test, dataset, classes[dataset_idx], transformed_cols[dataset_idx], verbose = 0, folder = "ica/", run_plot=True)
    km.run_kmeans()
    
    #random
    grp = RandomProjection(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "original/", run_plot=False)
    x_grp, x_test_grp = grp.run()
    km = K_Means(x_grp, y_train, x_test_grp, y_test, dataset, classes[dataset_idx], transformed_cols[dataset_idx], verbose = 0, folder = "grp/", run_plot=True)
    km.run_kmeans()
    
    #random forest
    rf = RandomForest(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "original/", run_plot=False)
    x_rf, x_test_rf = rf.run()
    km = K_Means(x_rf, y_train, x_test_rf, y_test, dataset, classes[dataset_idx], transformed_cols[dataset_idx], verbose = 0, folder = "rf/", run_plot=True)
    km.run_kmeans()


def run_EM_with_dim_reduction():
    #pca
    pca = PrincipalComponenetAnalysis(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "original/", run_plot=False)
    x_pca, x_test_pca = pca.run()
    em = EM(x_pca, y_train, x_test_pca, y_test, dataset, classes[dataset_idx], transformed_cols[dataset_idx], verbose = 0, folder = "pca/", run_plot=True)
    em.run_EM()
    
    #ica
    ica = IndependentComponentAnalysis(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "original/", run_plot=False)
    x_ica, x_test_ica = ica.run()
    em = EM(x_ica, y_train, x_test_ica, y_test, dataset, classes[dataset_idx], transformed_cols[dataset_idx], verbose = 0, folder = "ica/", run_plot=True)
    em.run_EM()
    
    #random
    grp = RandomProjection(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "original/", run_plot=False)
    x_grp, x_test_grp = grp.run()
    em = EM(x_grp, y_train, x_test_grp, y_test, dataset, classes[dataset_idx], transformed_cols[dataset_idx], verbose = 0, folder = "grp/", run_plot=True)
    em.run_EM()
    
    #random forest
    rf = RandomForest(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "original/", run_plot=False)
    x_rf, x_test_rf = rf.run()
    em = EM(x_rf, y_train, x_test_rf, y_test, dataset, classes[dataset_idx], transformed_cols[dataset_idx], verbose = 0, folder = "rf/", run_plot=True)
    em.run_EM()
    
    
    
def run_EM():
    em = EM(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], transformed_cols[dataset_idx], verbose = 0, run_plot=True)
    em.run_EM()
    

def run_pca():
    pca = PrincipalComponenetAnalysis(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "original/")
    pca.run()
    
def run_ica():
    ica = IndependentComponentAnalysis(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "original/")
    ica.run()
    
def run_grp():
    grp = RandomProjection(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "original/")
    grp.run()
  
def run_rf():
    rf = RandomForest(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "original/")
    rf.run()
   
def run_nn_dim_red():
    model_names = ["./models/neuralnet_bc.pkl", "./models/neuralnet_titanic.pkl"]
    
    def run_nn(reducer, folder):
        x_r, x_test_r = reducer.run()
        nn = NeuralNetwork(x_r, y_train, x_test_r, y_test, dataset_idx, folder = folder)
        nn.train_ANN(model_names[dataset_idx])
        nn.test_ANN(model_names[dataset_idx])
        
        
    pca = PrincipalComponenetAnalysis(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "pca/", run_plot=False)
    run_nn(pca, "pca/")
    
    ica = IndependentComponentAnalysis(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "ica/", run_plot=False)
    run_nn(ica, "ica/")

    grp = RandomProjection(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "grp/", run_plot=False)
    run_nn(grp, "grp/")

    rf = RandomForest(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "rf/", run_plot=False)
    run_nn(rf, "rf/")
    
def run_nn_dim_red_with_clustering():
    model_names = ["./models/neuralnet_bc.pkl", "./models/neuralnet_titanic.pkl"]
    
    def run_nn(reducer, folder):
        x_r, x_test_r = reducer.run()
        km = K_Means(x_r, y_train, x_test_r, y_test, dataset, classes[dataset_idx], transformed_cols[dataset_idx], verbose = 0, folder = folder)
        km_labels_x, km_labels_x_test = km.run_kmeans()
        '''print("train", x_train.shape, x_test.shape)
        print("red", x_r.shape, x_test_r.shape)
        print("label", km_labels_x.shape, km_labels_x_test.shape)'''

        em = EM(x_r, y_train, x_test_r, y_test, dataset, classes[dataset_idx], transformed_cols[dataset_idx], verbose = 0, folder = folder)
        em_labels_x, em_labels_x_test = em.run_EM()
        x_r = np.c_[x_r, km_labels_x]
        x_r = np.c_[x_r, em_labels_x]
        x_test_r = np.c_[x_test_r, km_labels_x_test]
        x_test_r = np.c_[x_test_r, em_labels_x_test]
        nn = NeuralNetwork(x_r, y_train, x_test_r, y_test, dataset_idx, folder = "clustering/"+folder)
        nn.train_ANN(model_names[dataset_idx])
        nn.test_ANN(model_names[dataset_idx])
        
        
    #pca = PrincipalComponenetAnalysis(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "pca/", run_plot=False)
    #run_nn(pca, "pca/")
    
    ica = IndependentComponentAnalysis(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "ica/", run_plot=False)
    run_nn(ica, "ica/")

    grp = RandomProjection(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "grp/", run_plot=False)
    run_nn(grp, "grp/")

    rf = RandomForest(x_train, y_train, x_test, y_test, dataset, classes[dataset_idx], verbose = 0, folder = "rf/", run_plot=False)
    run_nn(rf, "rf/")
   
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
    print(("YTrain Info: \n{0} \n").format(y_train.value_counts()))
    
    '''Experiments runs'''
    run_kmeans()
    run_EM()
    run_pca()
    run_ica()
    run_grp()
    run_rf()
    
    run_kmeans_with_dim_reduction()
    run_EM_with_dim_reduction()
    run_nn_dim_red()
    run_nn_dim_red_with_clustering()