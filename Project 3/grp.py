from sklearn.random_projection import GaussianRandomProjection as GRP
from matplotlib import pyplot as plt
import time

from numpy import linalg as LA
from numpy.linalg import eig
from scipy.stats import kurtosis


import seaborn as sns
from base import *

class RandomProjection():
    def __init__(self, x, y, x_test, y_test, dataset, classes, verbose = 0, folder = "original/", run_plot=True):
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        self.dataset = dataset
        self.verbose = verbose
        self.classes = classes
        self.folder = folder
        self.run_plot = run_plot

    def run(self):
        self.verbose = 0
        
        components = len(self.x.columns)
        avg_kurtosis = []
        
        for i in range(2, components+1):
            grp = GRP(random_state=10, n_components=i)
            x_grp = grp.fit_transform(self.x)
            kurt = abs(kurtosis(x_grp, axis=0)).mean()
            avg_kurtosis.append(kurt)
          
        if self.run_plot: 
            self.plot_kurtosis(avg_kurtosis)
        best_components = np.argmax(avg_kurtosis)+2
        return self.best_grp(best_components)
  

    def best_grp(self, best_components):
        start = time.time()
        grp = GRP(n_components=best_components, random_state=10)
        self.x_grp = grp.fit_transform(self.x)

        self.x_test_grp = grp.transform(self.x_test)
        
        if self.run_plot: 
            print("Best Components: ", best_components)
            print("Time GRP: ", time.time() - start)
            print("Mean Reconstruction Error: ",self.mean_reconstruction_error(grp, self.x_grp))
            self.plot_2D(self.x_grp, self.y, self.calculate_eigenvalues(self.x_grp))
            self.plot_3D(self.x_grp, self.y)
            self.plot_heat_map(grp)
            
        return self.x_grp, self.x_test_grp
 
        
    def mean_reconstruction_error(self, grp, x_grp):
        "ref https://www.kaggle.com/code/ericlikedata/reconstruct-error-of-pca/notebook"
        grp_proj_back=grp.inverse_transform(x_grp)
        total_loss=LA.norm((self.x-grp_proj_back),None)
        return total_loss
    
     
    def calculate_eigenvalues(self, x_grp):
        cov_matrix = np.cov(x_grp.T)
        eigenvalues, eigenvectors = eig(cov_matrix)
        return eigenvalues
           
    def plot_kurtosis(self, avg_kurtosis):
        _, ax = plt.subplots()
        y = avg_kurtosis
        xi = np.arange(2, len(y)+2, step=1)

        #plt.ylim(0.0,1.1)
        plt.plot(xi, y, marker='o', linestyle='--', color='b')

        plt.xlabel('Number of Components')
        plt.xticks(np.arange(2, len(y)+2, step=1)) 
        plt.ylabel('Average Kurtosis')
        plt.title('The average kurtosis over the number of components')

        plt.axhline(y=np.max(y), color='r', linestyle='-')
        plt.text(3, np.max(y) -5, 'Max Kurtosis', color = 'red', fontsize=16)

        ax.grid(axis='x')
        plt.savefig("./images/"+self.dataset+"/grp/"+self.folder+"kurtosis")
        plt.clf()
        
        
    def plot_2D(self, x_grp, y, eigenvals):
        y = y.replace({0:self.classes[0], 1:self.classes[1]})
        ax = sns.scatterplot(x=x_grp[:,0], y=x_grp[:,1], hue=y, palette ='Set1' )
        ax.axvline(0, c="w", lw=1, ls="--"), ax.axhline(0, c="w", lw=1, ls="--")
        ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_title("Transformed Dataset")
        ax.arrow(0, 0, -eigenvals[0], 0, color="green", width=0.1)
        ax.arrow(0, 0, 0, -eigenvals[1], color="green", width=0.1)
        ax.set_facecolor('xkcd:black')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.savefig("./images/"+self.dataset+"/grp/"+self.folder+"2D")
        plt.clf()
        
    def plot_3D(self, x_grp, y):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_grp[:,0], x_grp[:,1], x_grp[:,2], c=y, s=60)
        ax.legend(self.classes)
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Third Principal Component')
        ax.view_init(30, 120)
        plt.savefig("./images/"+self.dataset+"/grp/"+self.folder+"3D")
        plt.clf()
        
    def plot_heat_map(self, grp):
        df_pc = pd.DataFrame(grp.components_, columns = self.x.columns)
        plt.figure(figsize=(15, 14))
        ax = sns.heatmap(df_pc, cmap='YlGnBu')
        ax.set_aspect("equal")
        plt.title('Principal Components correlation with the features')
        plt.xlabel('Features')
        plt.ylabel('Principal Components')
        plt.savefig("./images/"+self.dataset+"/grp/"+self.folder+"heat_map")
        plt.clf()