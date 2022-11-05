from sklearn.mixture import GaussianMixture as ExpectationMaximization
from sklearn.metrics import silhouette_score, homogeneity_score, silhouette_samples
from sklearn.manifold import TSNE

from scipy.spatial.distance import cdist
from matplotlib.legend_handler import HandlerTuple

import itertools
from scipy import linalg

from base import *

class EM():
    def __init__(self, x, y, x_test, y_test, dataset, classes, transformed_cols, verbose = 0, folder = "original/", run_plot= False):
        self.num_clusters = 15
        self.cv_types = ["spherical", "tied", "diag", "full"]
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        self.dataset = dataset
        self.verbose = verbose
        self.aic, self.bic= ([[] for _ in range(len(self.cv_types))] for _ in range(2))
        self.score, self.sil, self.sample_sil, self.hmg, self.time \
            = ([[] for _ in range(len(self.cv_types))] for _ in range(5))
                    
        self.f1, self.train_accuracy, self.precision, self.recall \
            = ([[] for _ in range(len(self.cv_types))] for _ in range(4))
            
        self.folder = folder
        self.transformed_cols = transformed_cols
        self.transformed_x = self.get_transformed(self.x)
        self.transformed_x_test = self.get_transformed(self.x_test)
        self.classes = classes
        self.map = self.get_cmap(self.num_clusters)
        self.colors = [self.map(i) for i in range(self.num_clusters)]
        self.run_plot = run_plot
  
    def run_EM(self):
        self.verbose = 0
        
        clusters = np.arange(2, self.num_clusters+1)
        lowest_bic = np.infty; best_cv_idx = 0; best_k = 1;
                
        for i, cv_type in enumerate(self.cv_types):
            for k in clusters:
                em, _ = self.EM(k, cv_type, i)
                if self.bic[i][-1] < lowest_bic:
                    lowest_bic = self.bic[i][-1]
                    best_gmm = em; best_k = k; best_cv_idx = i
            
        if self.run_plot:                
            self.plot_score_chart('logprob', 'The Log Probability', self.score)
            #self.plot_score_chart('distortion', 'Distortion: Average Error from the center', self.distortion)   
            self.plot_score_chart('sil', 'Silhouette Score', self.sil)   
            self.plot_score_chart('hmg', 'Homogenous Score', self.hmg)   
            self.plot_score_chart('aic', 'Akaike Information', self.aic)   
            self.plot_score_chart('bic', 'Bayesian Information', self.bic)
            self.plot_time()
          
            self.plot_bic(self.bic, best_gmm, self.num_clusters)  
        
        print(("Best GMM: {0} Best k: {1}\n").format(best_gmm.covariance_type, best_k))
        best = self.best_cluster(best_k, best_gmm.covariance_type, best_cv_idx)
        return best.labels_, best.predict(self.x_test)

     
    def get_accuracies(self, em):
        y_train_pred = em.predict(self.x)
        train_accuracy = np.mean(y_train_pred.ravel() == self.y.ravel()) * 100

        y_test_pred = em.predict(self.x_test)
        test_accuracy = np.mean(y_test_pred.ravel() == self.y_test.ravel()) * 100
        return (train_accuracy, test_accuracy)
        
    def best_cluster(self, best_k, cv_type, cv_idx):
        self.verbose = 1
        em, y_labels = self.EM(best_k, cv_type, cv_idx)
        y_pred = self.predict(self.y, y_labels, best_k)
        
        if self.run_plot:
            self.plot_clusters_labels(em, y_labels, best_k)
            self.plot_clusters_pred(y_pred, best_k)
            self.plot_distribution(y_labels, best_k)
            self.plot_silhouette(best_k, y_labels, cv_idx)
            self.plot_accuracies(best_k)
            self.different_init(best_k)
        return em
    
    def predict(self, y_true, y_labels, k):
        y_pred = np.empty_like(y_true)
        
        for c in range(k):
            c_idxs = y_labels == c
            cluster_y = y_true[c_idxs]
            if (len(cluster_y) !=0):
                y_pred[c_idxs] = np.argmax(np.bincount(cluster_y))
            
        return y_pred
  
    def test(self, y, y_labels, k):
        y_predict = self.predict(y, y_labels, k)
        return metrics(y, y_predict, self.dataset, "/EM/"+self.folder+str(k), self.verbose)
    
    def EM(self, k, cv_type, cv_idx):
        em = ExpectationMaximization(covariance_type=cv_type, init_params="k-means++", warm_start=True, n_components=k,
                                random_state= 10,
                                n_init = 5,
                                verbose = self.verbose)
        start_time = time.time()
        em.fit(self.x)
        time_to_train = time.time() - start_time
        y_labels = em.predict(self.x)
        
                
        self.time[cv_idx].append(time_to_train)
        self.sil[cv_idx].append(silhouette_score(self.x, y_labels))
        self.sample_sil[cv_idx].append(silhouette_samples(self.x, y_labels))
        self.hmg[cv_idx].append(homogeneity_score(self.y, y_labels))
        
        self.score[cv_idx].append(em.score(self.x))
        self.aic[cv_idx].append(em.aic(self.x))
        self.bic[cv_idx].append(em.bic(self.x))

        f1, accuracy, precision, recall = self.test(self.y, y_labels, k)
        self.f1[cv_idx].append(f1); self.train_accuracy[cv_idx].append(accuracy);
        self.precision[cv_idx].append(precision); self.recall[cv_idx].append(recall)
        
        if self.verbose:
            print(("Time to Train: {0} \n").format(time_to_train))
            print(("Score: {0} \n").format(self.score[cv_idx][-1]))
            print(("HMG: {0} \n").format(self.hmg[cv_idx][-1]))
            print(("SIL: {0} \n").format(self.sil[cv_idx][-1]))
            
        return em, y_labels
    
    def plot_time(self):
        plt.title("Time to fit the clusters")
        plt.xlabel('Number of clusters')
        plt.ylabel('Time')
        
        for i, cv_type in enumerate(self.cv_types):
            plt.plot(np.arange(2, self.num_clusters+1), self.time[i], label = cv_type)
          
        plt.legend(loc = 'best')     
        plt.savefig("./images/"+self.dataset+"/EM/"+self.folder+"time")
        plt.clf()
                
    def plot_score_chart(self, score_name, title, score):
        plt.title(title)
        plt.xlabel('Number of clusters')
        plt.ylabel('score')
        for i, cv_type in enumerate(self.cv_types):
            plt.plot(np.arange(2, self.num_clusters+1), score[i], label = cv_type)
        
        plt.legend(loc = 'best')   
        plt.savefig("./images/"+self.dataset+"/EM/"+self.folder+score_name)
        plt.clf()
      
    
    def get_initial_means(self, X, init_params, k):
        # Run a GaussianMixture with max_iter=0 to output the initalization means
        gmm = ExpectationMaximization(
            n_components=k, init_params=init_params, tol=1e-9, max_iter=0, random_state=10
        ).fit(X)
        return gmm.means_
   
    def different_init(self, best_k):
        methods = ["kmeans", "random_from_data", "k-means++", "random"]
        times_init = {}
        relative_times = {}

        #plt.figure(figsize=(4 * len(methods) // 2, 6))
        plt.subplots_adjust(
            bottom=0.1, hspace=0.15, wspace=0.05, left=0.05, right=0.95
        )

        for n, method in enumerate(methods):
            plt.subplot(2, len(methods) //2, n + 1)

            start = time.time()
            ini = self.get_initial_means(self.x, method, best_k)
            init_time = time.time() - start

            gmm = ExpectationMaximization(
                n_components=best_k, means_init=ini, n_init=5, max_iter=300, random_state=10, warm_start=True,
            ).fit(self.x)

            times_init[method] = init_time
            y_labels = gmm.predict(self.x)
            for i in range(best_k):
                #data = self.x[y_labels == i]
                transformed_data = self.transformed_x.iloc[y_labels == i]
                plt.scatter(transformed_data.iloc[:, 0], transformed_data.iloc[:, 1], marker="x")
                centers = transformed_data.mean(axis=0)
                plt.scatter(centers[0], centers[1], s=75, marker="D", c="orange", lw=1.5, edgecolors="black")
                
            '''plt.scatter(
                ini[:, 0], ini[:, 1], s=75, marker="D", c="orange", lw=1.5, edgecolors="black"
            )'''
            relative_times[method] = times_init[method] / times_init[methods[0]]

            plt.xticks(())
            plt.yticks(())
            plt.title(method, loc="left", fontsize=12)
            plt.title(
                "Iter %i | Init Time %.2fx" % (gmm.n_iter_, relative_times[method]),
                loc="right",
                fontsize=7,
            )
        plt.suptitle("GMM iterations and relative time taken to initialize Best K= "+str(best_k))
        plt.savefig("./images/"+self.dataset+"/EM/"+self.folder+"/best/"+"init_times")
        plt.clf()
      
    def make_ellipses(self, gmm, ax, best_k):
        for n in range(best_k):
            if gmm.covariance_type == "full":
                covariances = gmm.covariances_[n][:2, :2]
            elif gmm.covariance_type == "tied":
                covariances = gmm.covariances_[:2, :2]
            elif gmm.covariance_type == "diag":
                covariances = np.diag(gmm.covariances_[n][:2])
            elif gmm.covariance_type == "spherical":
                covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            ell = matplotlib.patches.Ellipse(
                gmm.means_[n, :2], v[0], v[1], 180 + angle, color=self.colors[n]
            )
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(1)
            ax.add_artist(ell)
            ax.set_aspect("equal", "datalim")
          
    def plot_accuracies(self, best_k):
        "ref: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py"
                
        estimators = {
            cov_type: ExpectationMaximization(
                n_components=best_k, covariance_type=cov_type, n_init=5, max_iter=20, random_state=10
            )
            for cov_type in self.cv_types
        }

        n_estimators = len(estimators)
        _, axes = plt.subplots(2, n_estimators // 2)

        for index, (name, estimator) in enumerate(estimators.items()):
             # Since we have class labels for the training data, we can
            # initialize the GMM parameters in a supervised manner.
            '''estimator.means_init = np.array(
                [self.x[self.y == i].mean(axis=0) for i in range(best_k)])'''
            # Train the other parameters using the EM algorithm.
            estimator.fit(self.x)
            
            h = plt.subplot(2, n_estimators // 2, index + 1)
            self.make_ellipses(estimator, h, best_k)
               
            y_labels = estimator.predict(self.x) 
            for i in range(best_k):
                transformed_data = self.transformed_x.iloc[y_labels == i]
                plt.scatter(transformed_data.iloc[:, 0], transformed_data.iloc[:, 1], s=1.1, color=self.colors[i], label=i)
               
            y_labels_test = estimator.predict(self.x_test)  
            # Plot the test data with crosses
            for i in range(best_k):
                transformed_data = self.transformed_x_test.iloc[y_labels_test == i]
                plt.scatter(transformed_data.iloc[:, 0], transformed_data.iloc[:, 1], marker="x", color=self.colors[i], label=i)

            y_train_pred = estimator.predict(self.x)
            y_train_pred = self.predict(self.y, y_train_pred, best_k)
            train_accuracy = np.mean(y_train_pred.ravel() == self.y.ravel()) * 100
            plt.text(0.05, 0.9, "Train accuracy: %.1f" % train_accuracy, transform=h.transAxes)

            y_test_pred = estimator.predict(self.x_test)
            y_test_pred = self.predict(self.y_test, y_test_pred, best_k)
            test_accuracy = np.mean(y_test_pred.ravel() == self.y_test.ravel()) * 100
            plt.text(0.05, 0.8, "Test accuracy: %.1f" % test_accuracy, transform=h.transAxes)

            plt.xticks(())
            plt.yticks(())
            plt.title(name)

        handles1, labels1 = axes[0][0].get_legend_handles_labels()
        handles2, labels2 = axes[0][1].get_legend_handles_labels()
        handles1, labels1 = handles1[:best_k], labels1[:best_k]
        handles2, labels2 = handles2[best_k:2*best_k], labels2[best_k:2*best_k]
        plt.legend([(handles1[idx], handles2[idx]) for idx in range(best_k)],\
            [labels1[idx] for idx in range(best_k)], \
            handler_map={tuple: HandlerTuple(ndivide=2)}, \
            loc="lower right", frameon=True, ncol=3, scatterpoints=1)\
            .get_frame().set_edgecolor('black')
        plt.subplots_adjust(
            bottom=0.01, hspace=0.15, wspace=0.05, left=0.01, right=0.99
        )
        plt.suptitle("Train and Test accuracies for different covariances Best K= "+str(best_k))
        plt.savefig("./images/"+self.dataset+"/EM/"+self.folder+"/best/"+"accuracies")
        plt.clf()
 
        
    def plot_bic(self, bic, best_gmm, k):
        "ref: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py"
        bic = np.array(bic)
        color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
        bars = []
        # Plot the BIC scores
        #plt.figure(figsize=(8, 6))
        #spl = plt.subplot(1, 1, 1)
        print("bic", bic)
        for i, (cv_type, color) in enumerate(zip(self.cv_types, color_iter)):
            b = bic[i]
            xpos = np.arange(2, len(b)+2) + 0.2 * (i - 2)

            bars.append(
                plt.bar(
                    xpos,
                    b,
                    width=0.2,
                    color=color,
                    label = cv_type
                )
            )
        plt.xticks(np.arange(2, k+1))
        plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
        plt.title("BIC score per model")
        xpos = (
            np.mod(bic.argmin(), k-1)
            + 0.2 * (np.floor(bic.argmin() / (k-1)) -2)
        )+2-0.2
        print("min", bic.argmin(), np.mod(bic.argmin(), k-1), np.floor(bic.argmin() / (k-1)), xpos)
        plt.text(xpos, bic.min(), "*", fontsize=14)
        plt.xlabel("Number of components")
        
        handles = []; labels = [];
        for i, b in enumerate(bars):
            if (len(b.get_children()) > 0):
                handles += [b[0]]
                labels += [self.cv_types[i]]
        
        plt.legend(handles, labels)

        # Plot the winner
        '''splot = plt.subplot(2, 1, 2)
        Y_ = clf.predict(self.x)
        for i, color in enumerate(color_iter):
            if not np.any(Y_ == i):
                continue
            data = self.transformed_x[Y_ == i]
            cov = np.cov(data.iloc[:, 0], data.iloc[:, 1])
            mean = (np.mean(data.iloc[:, 0]), np.mean(data.iloc[:, 1]))
            
            cov = np.nan_to_num(cov)
            v, w = linalg.eigh(cov)
            
            plt.scatter(data.iloc[:, 0], data.iloc[:, 1], 0.8, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan2(w[0][1], w[0][0])
            angle = 180.0 * angle / np.pi  # convert to degrees
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            ell = matplotlib.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)

        plt.xticks(())
        plt.yticks(())
        plt.title(
            f"Selected GMM: {best_gmm.covariance_type} model, "
            f"{best_gmm.n_components} components"
        )
        plt.subplots_adjust(hspace=0.35, bottom=0.02)'''
        plt.savefig("./images/"+self.dataset+"/EM/"+self.folder+"all_bic")
        plt.clf()
        
        
    def plot_clusters_labels(self, km, k_labels, best_k):
        x_transformed = self.transformed_x
        
        #centers = km.cluster_centers_ 
        plt.scatter(x_transformed.iloc[:,0], x_transformed.iloc[:,1],  c=k_labels, cmap = "jet", edgecolor = "None", alpha=0.35)
        
        '''plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",)'''
        
        for i in range(best_k):
            #data = self.x[y_labels == i]
            transformed_data = self.transformed_x.iloc[k_labels == i]
            centers = transformed_data.mean(axis=0)
            plt.scatter(
                centers[0],
                centers[1],
                marker="o",
                c="white",
                alpha=1,
                s=200,
                edgecolor="k",)
            plt.scatter(centers[0], centers[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k", label = "K="+str(i))
            
        '''for i, center in enumerate(centers):    
            plt.scatter(center[0], center[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k", label = "K="+str(i))'''
        
        plt.title('kMeans Clusters with Best K= '+ str(best_k))
        plt.xlabel("Transformed Feature 1")
        plt.ylabel("Transformed Feature 2")
        #plt.legend(loc = 'best')
        plt.savefig("./images/"+self.dataset+"/EM/"+self.folder+"/best/"+"clusters_labels")
        plt.clf()
     
    def plot_clusters_pred(self, y_pred, best_k):
        _, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        
        for i, class_ in enumerate(self.classes):
            transformed_data = self.transformed_x.iloc[y_pred == i]
            ax1.scatter(transformed_data.iloc[:,0],transformed_data.iloc[:,1], cmap = "jet", edgecolor = "None", alpha=0.35, label=class_)
        ax1.set_title('EM Prediction')
        
        for i, class_ in enumerate(self.classes):
            y = self.y.to_numpy()
            transformed_data = self.transformed_x.iloc[y == i]
            ax2.scatter(transformed_data.iloc[:,0],transformed_data.iloc[:,1], cmap = "jet", edgecolor = "None", alpha=0.35, label=class_)
        ax2.set_title('Original Labels')
        
        plt.xlabel("Transformed Feature 1")
        plt.ylabel("Transformed Feature 2")
        plt.suptitle("Predictions Vs the Original lables clustering Best K= "+str(best_k))
        plt.legend(loc = 'best')
        plt.savefig("./images/"+self.dataset+"/EM/"+self.folder+"/best/"+"clusters_pred")
        plt.clf()

    def plot_distribution(self, y_labels, k):
        plt.title("Distribution of dataset K= "+str(k))
        plt.xlabel("Clusters")
        plt.ylabel("Count per cluster") 
        plt.hist(y_labels, bins = np.arange(k)-0.5, rwidth = 0.5)
        plt.xticks(np.arange(k))
        plt.savefig("./images/"+self.dataset+"/EM/"+self.folder+"/best/"+"distribution")
        plt.clf()
        
    def plot_silhouette(self, k, y_labels, cv_idx):
        "ref: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html"
        
        y_lower = 10
        average = self.sil[cv_idx][k-2]
        sample = self.sample_sil[cv_idx][k-2]
        
        for i in range(k):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            idxs = y_labels == i
            ith_cluster_silhouette_values = sample[idxs]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            plt.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        plt.title("The silhouette plot for the various clusters. Best K= "+str(k))
        plt.xlabel("The silhouette coefficient values")
        plt.ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        plt.axvline(x=average, color="red", linestyle="--")
        plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.savefig("./images/"+self.dataset+"/EM/"+self.folder+"/best/"+"sil_samples")
        plt.clf()        
         
              
    def get_transformed(self, data):
        #Creating a 2D visualization
        tsne = TSNE(verbose= self.verbose, perplexity=40, n_iter= 300)
        return pd.DataFrame(tsne.fit_transform(data))
        #return data.loc[:, self.transformed_cols]
    
    def get_cmap(self,n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)  
        

        
    