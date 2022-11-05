from sklearn.decomposition import PCA

from numpy import linalg as LA
import seaborn as sns
from base import *

class PrincipalComponenetAnalysis():
    def __init__(self, x, y, x_test, y_test, dataset, classes, verbose = 0, folder = "original/", run_plot = True):
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
        
        start = time.time()
        pca = PCA(svd_solver="full", random_state=10)
        self.x_pca = pca.fit_transform(self.x)
        print("Time PCA: ", time.time() - start)
        
        principalDf = pd.DataFrame(data = self.x_pca[:,:2]
            , columns = ['principal component 1', 'principal component 2'])
        
        self.x_test_pca = pca.transform(self.x_test)
        self.eigenvecs = pca.components_
        self.eigenvals = pca.explained_variance_
        if self.run_plot:
            print("Mean Reconstruction Error: ", self.mean_reconstruction_error(pca, self.x_pca))
            self.plot_variances(pca)
            self.plot_cum_variance(pca)
            self.plot_2D(self.x_pca, self.y, self.eigenvals)
            self.plot_3D(self.x_pca, self.y)
            self.plot_heat_map(pca)
        return self.x_pca, self.x_test_pca
 
    def mean_reconstruction_error(self, pca, x_pca):
        "ref https://www.kaggle.com/code/ericlikedata/reconstruct-error-of-pca/notebook"
        pca_proj_back=pca.inverse_transform(x_pca)
        total_loss=LA.norm((self.x-pca_proj_back),None)
        return total_loss
    
    def plot_variances(self, pca):
        "ref https://vitalflux.com/pca-explained-variance-concept-python-example/"
        exp_var_pca = pca.explained_variance_ratio_
        cum_sum_eigenvalues = np.cumsum(exp_var_pca)
        plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Explained Individual variance')
        plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Explained Cumulative variance')
        plt.ylabel('Explained Variance ratio')
        plt.xlabel('Principal component index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig("./images/"+self.dataset+"/pca/"+self.folder+"variances")
        plt.clf()
      
        
    def plot_cum_variance(self, pca):
        "ref: https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/"
        _, ax = plt.subplots()
        y = np.cumsum(pca.explained_variance_ratio_)
        xi = np.arange(1, len(y)+1, step=1)

        plt.ylim(0.0,1.1)
        plt.plot(xi, y, marker='o', linestyle='--', color='b')

        plt.xlabel('Number of Components')
        plt.xticks(np.arange(0, len(y)+1, step=1)) 
        plt.ylabel('Cumulative variance (%)')
        plt.title('The number of components needed to explain variance')

        plt.axhline(y=0.95, color='r', linestyle='-')
        plt.text(0.5, 0.85, '95% cut off threshold', color = 'red', fontsize=16)

        ax.grid(axis='x')
        plt.savefig("./images/"+self.dataset+"/pca/"+self.folder+"cum_variances")
        plt.clf()
 
    def plot_2D(self, x_pca, y, eigenvals):
        y = y.replace({0:self.classes[0], 1:self.classes[1]})
        ax = sns.scatterplot(x=x_pca[:,0], y=x_pca[:,1], hue=y, palette ='Set1' )
        ax.axvline(0, c="w", lw=1, ls="--"), ax.axhline(0, c="w", lw=1, ls="--")
        ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_title("Transformed Dataset")
        ax.arrow(0, 0, -eigenvals[0], 0, color="green", width=0.1)
        ax.arrow(0, 0, 0, -eigenvals[1], color="green", width=0.1)
        ax.set_facecolor('xkcd:black')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.savefig("./images/"+self.dataset+"/pca/"+self.folder+"2D")
        plt.clf()
        
    def plot_3D(self, x_pca, y):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_pca[:,0], x_pca[:,1], x_pca[:,2], c=y, s=60)
        ax.legend(self.classes)
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Third Principal Component')
        ax.view_init(30, 120)
        plt.savefig("./images/"+self.dataset+"/pca/"+self.folder+"3D")
        plt.clf()
        
    def plot_heat_map(self, pca):
        df_pc = pd.DataFrame(pca.components_, columns = self.x.columns)
        plt.figure(figsize=(15, 14))
        ax = sns.heatmap(df_pc, cmap='YlGnBu')
        ax.set_aspect("equal")
        plt.title('Principal Components correlation with the features')
        plt.xlabel('Features')
        plt.ylabel('Principal Components')
        plt.savefig("./images/"+self.dataset+"/pca/"+self.folder+"heat_map")
        plt.clf()