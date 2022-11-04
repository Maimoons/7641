from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

from numpy.linalg import eig
import seaborn as sns
from base import *

class RandomForest():
    def __init__(self, x, y, x_test, y_test, dataset, classes, verbose = 0, folder = "original/"):
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        self.dataset = dataset
        self.verbose = verbose
        self.classes = classes
        self.folder = folder

    def run_rf(self):
        self.verbose = 0
        
        rf = RandomForestClassifier(random_state=10, n_estimators=10)
        start = time.time()
        rfecv = RFECV(rf,cv=5,scoring="neg_mean_squared_error")
        rfecv.fit(self.x,self.y)
        print("Time RF: ", time.time() - start)
        
        selected_features = np.array(self.x.columns)[rfecv.get_support()]
        print("Selected Features: ", len(selected_features), selected_features)
        self.x_rf = self.x[selected_features]
        self.x_test_rf = self.x_test[selected_features]


        importances =  rfecv.estimator_.feature_importances_ 
        indices = np.argsort(importances)[::-1]
        
        self.plot_2D(self.x_rf, self.y, self.calculate_eigenvalues(self.x_rf))
        self.plot_3D(self.x, self.y)
        self.variable_importance_plot(importances, indices, selected_features)
 
    def calculate_eigenvalues(self, x_rf):
        cov_matrix = np.cov(x_rf.T)
        eigenvalues, eigenvectors = eig(cov_matrix)
        return eigenvalues
    
    def variable_importance_plot(self, importance, indices, names_index):
        index = np.arange(len(names_index))

        feature_space = []

        for i in range(indices.shape[0] - 1, -1, -1):
            feature_space.append(names_index[indices[i]])

        plt.subplots(figsize=(15, 10))
        importance_desc = sorted(importance)
        plt.barh(index,
                importance_desc,
                align="center",
                color = '#FFB6C1')
        
        plt.yticks(index,feature_space)
        plt.xlim(0, max(importance_desc) + 0.01)
        plt.title('Feature importances with Random Forest')
        plt.xlabel('Mean Decrease in Impurity')
        plt.ylabel('Features')
        plt.savefig("./images/"+self.dataset+"/rf/"+self.folder+"importance")
        plt.clf()
 
    def plot_2D(self, x_rf, y, eigenvals):
        y = y.replace({0:self.classes[0], 1:self.classes[1]})
        ax = sns.scatterplot(x=x_rf.iloc[:,0], y=x_rf.iloc[:,1], hue=y, palette ='Set1' )
        ax.axvline(0, c="w", lw=1, ls="--"), ax.axhline(0, c="w", lw=1, ls="--")
        ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_title("Transformed Dataset")
        ax.arrow(0, 0, -eigenvals[0], 0, color="green", width=0.1)
        ax.arrow(0, 0, 0, -eigenvals[1], color="green", width=0.1)
        ax.set_facecolor('xkcd:black')
        plt.xlabel('First Feature')
        plt.ylabel('Second Feature')
        plt.savefig("./images/"+self.dataset+"/rf/"+self.folder+"2D")
        plt.clf()
        
    def plot_3D(self, x_rf, y):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_rf.iloc[:,0], x_rf.iloc[:,1], x_rf.iloc[:,2], c=y, s=60)
        ax.legend(self.classes)
        ax.set_xlabel('First Feature')
        ax.set_ylabel('Second Feature')
        ax.set_zlabel('Third Feature')
        ax.view_init(30, 120)
        plt.savefig("./images/"+self.dataset+"/rf/"+self.folder+"3D")
        plt.clf()
