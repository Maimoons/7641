from statistics import covariance
from sklearn.cluster import KMeans, BisectingKMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, homogeneity_score, silhouette_samples
from sklearn.manifold import TSNE

from scipy.spatial.distance import cdist
from matplotlib.legend_handler import HandlerTuple

from base import *

class K_Means():
    def __init__(self, x, y, x_test, y_test, dataset, classes, transformed_columns, verbose = 0, folder = ""):
        self.num_clusters = 15
        self.algorithms = [KMeans, BisectingKMeans]
        self.algorithms_n = ['KMeans', 'BisectingKMeans']
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        self.dataset = dataset
        self.verbose = verbose
        self.wcss, self.distortion, self.sil, self.sample_sil, self.hmg, self.time \
            = ([[] for _ in range(len(self.algorithms))] for _ in range(6))
            
        self.f1, self.train_accuracy, self.precision, self.recall \
            = ([[] for _ in range(len(self.algorithms))] for _ in range(4))
            
        self.folder = folder
        self.transformed_cols = transformed_columns
        self.transformed_x = self.get_transformed(self.x)
        self.transformed_x_test = self.get_transformed(self.x_test)
        self.classes = classes
        self.map = self.get_cmap(self.num_clusters)
        self.colors = [self.map(i) for i in range(self.num_clusters)]
           
    def run_kmeans(self):
        self.verbose = 0
        
        clusters = np.arange(2, self.num_clusters+1)
        lowest_sse = np.infty; best_sse_idx = 0; best_k = 1;

        for i, Algorithm in enumerate(self.algorithms):
            for k in clusters:
                km = self.k_means(k, Algorithm, i)
                
                if self.wcss[i][-1] < lowest_sse:
                    lowest_sse = self.wcss[i][-1]
                    best_km = Algorithm; best_sse_idx = i
         
        best_k = self.plot_elbow(best_km)
        self.plot_score_chart('wcss', 'The Elbow with WCSS', self.wcss)
        self.plot_score_chart('distortion', 'Distortion: Average Error from the center', self.distortion)   
        self.plot_score_chart('sil', 'Silhouette Score', self.sil)   
        self.plot_score_chart('hmg', 'Homogenous Score', self.hmg)   
        self.plot_time()
        
        print(("Best KM: {0} Best k: {1}\n").format(best_km, best_k))
        self.best_cluster(best_k, best_km, best_sse_idx)

    def plot_elbow(self, model):
        from yellowbrick.cluster import KElbowVisualizer
        ax = plt.subplot(1, 1, 1)
        visualizer = KElbowVisualizer(model(), ax, k=(2, self.num_clusters))
        visualizer.fit(self.x)        
        visualizer.show(outpath="./images/"+self.dataset+"/kmeans/"+self.folder+"/best/elbow_method")
        #plt.clf() 
        return visualizer.elbow_value_
          
    def best_cluster(self, best_k, Algorithm, algo_idx):
        self.verbose = 1
        km = self.k_means(best_k, Algorithm, algo_idx)
        y_labels = km.labels_
        y_pred = self.predict(self.y, y_labels, best_k)

        self.plot_clusters_labels(km, y_labels, best_k)
        self.plot_clusters_pred(y_pred, best_k)
        self.plot_distribution(y_labels, best_k)
        self.plot_silhouette(best_k, y_labels, algo_idx)
        self.plot_accuracies(best_k)
        self.different_init(best_k, Algorithm)
        
        # run test
        return km
    
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
        return metrics(y, y_predict, self.dataset, "/kmeans/"+self.folder+str(k), self.verbose)
    
    def k_means(self, k, Algorithm, algo_idx):
        km = Algorithm(init="k-means++", max_iter=300, n_clusters=k, n_init=5,
                                random_state= 10,
                                verbose = 0)
        start_time = time.time()
        km.fit(self.x)
        time_to_train = time.time() - start_time
        y_labels = km.predict(self.x)
        centers= km.cluster_centers_
        
        self.time[algo_idx].append(time_to_train)
        self.wcss[algo_idx].append(km.inertia_)
        self.sil[algo_idx].append(silhouette_score(self.x, y_labels))
        self.sample_sil[algo_idx].append(silhouette_samples(self.x, y_labels))
        self.hmg[algo_idx].append(homogeneity_score(self.y, y_labels))
        self.distortion[algo_idx].append(sum(np.min(cdist(self.x, centers,
                                        'euclidean'), axis=1)) / self.x.shape[0])
        
        f1, accuracy, precision, recall = self.test(self.y, y_labels, k)
        self.f1[algo_idx].append(f1); self.train_accuracy[algo_idx].append(accuracy);
        self.precision[algo_idx].append(precision); self.recall[algo_idx].append(recall)
        
        if self.verbose:
            print(("Time to Train: {0} \n").format(time_to_train))
            print(("WCSS: {0} \n").format(self.wcss[algo_idx][-1]))
            print(("Distortion: {0} \n").format(self.distortion[algo_idx][-1]))
            print(("HMG: {0} \n").format(self.hmg[algo_idx][-1]))
            print(("SIL: {0} \n").format(self.sil[algo_idx][-1]))
            
        return km
            
    def plot_score_chart(self, score_name, title, score):
        plt.title(title)
        plt.xlabel('Number of clusters')
        plt.ylabel('score')
        maximum = 0
        for i, algo in enumerate(self.algorithms):
            plt.plot(np.arange(2, self.num_clusters+1), score[i], label = self.algorithms_n[i])
            maximum = max(max(score[i]), maximum)
        
        plt.xticks(np.arange(2, self.num_clusters+1))
        #plt.yticks(np.arange(0, maximum, 500))
        plt.legend(loc = 'best')
        #plt.grid(True) 
        plt.savefig("./images/"+self.dataset+"/kmeans/"+self.folder+score_name)
        plt.clf()
       
    def plot_time(self):
        plt.title("Time to fit the clusters")
        plt.xlabel('Number of clusters')
        plt.ylabel('score')
        for i, algo in enumerate(self.algorithms):
            plt.plot(np.arange(2, self.num_clusters+1), self.time[i], label = self.algorithms_n[i])
          
        plt.legend(loc = 'best')     
        plt.savefig("./images/"+self.dataset+"/kmeans/"+self.folder+"time")
        plt.clf()
   
   
    def make_ellipses(self, covariances, mean, ax, color):
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = matplotlib.patches.Ellipse(
            mean, v[0], v[1], 180 + angle, color=color
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.35)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")
          
    def plot_accuracies(self, best_k):
        "ref: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py"
        estimators = {
            self.algorithms_n[i]: Algorithm(init="k-means++", max_iter=20, n_clusters=best_k, n_init=5,
                                random_state= 10,
                                verbose = 0)
            for i, Algorithm in enumerate(self.algorithms)
        }

        n_estimators = len(estimators)
        _, axes = plt.subplots(1, n_estimators)
        
        for index, (name, estimator) in enumerate(estimators.items()):
            # Train the other parameters using the algorithm.
            estimator.fit(self.x)
            h = plt.subplot(1, n_estimators, index + 1)
               
            y_labels = estimator.predict(self.x) 
            for i in range(best_k):
                transformed_data = self.transformed_x.iloc[y_labels == i]
                plt.scatter(transformed_data.iloc[:, 0], transformed_data.iloc[:, 1], s=1.1, color = self.colors[i], label=i)
                
                covariances = np.cov(transformed_data.iloc[:, 0], transformed_data.iloc[:, 1])
                mean = (np.mean(transformed_data.iloc[:, 0]), np.mean(transformed_data.iloc[:, 1]))
                self.make_ellipses(covariances, mean, h, self.colors[i])
            
            y_labels_test = estimator.predict(self.x_test)  
            # Plot the test data with crosses
            for i in range(best_k):
                transformed_data = self.transformed_x_test[y_labels_test == i]
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

        plt.suptitle("Train and Test accuracies for different covariances Best K= "+str(best_k))
        handles1, labels1 = axes[0].get_legend_handles_labels()
        handles2, labels2 = axes[1].get_legend_handles_labels()
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
        plt.savefig("./images/"+self.dataset+"/kmeans/"+self.folder+"/best/"+"accuracies")
        plt.clf()
 
    def different_init(self, best_k, Algorithm):
        methods = ["k-means++", "random"]
        times_init = {}
        relative_times = {}

        #plt.figure(figsize=(4 * len(methods) // 2, 6))
        plt.subplots_adjust(
            bottom=0.1, hspace=0.15, wspace=0.05, left=0.05, right=0.95
        )

        for n, method in enumerate(methods):
            plt.subplot(1, len(methods), n + 1)

            start = time.time()
            km = Algorithm(init=method, max_iter=300, n_clusters=best_k,
                                random_state= 10,
                                verbose = 0).fit(self.x)
            init_time = time.time() - start
            times_init[method] = init_time
            y_labels = km.predict(self.x)
            for i in range(best_k):
                #data = self.x[y_labels == i]
                transformed_data = self.transformed_x.iloc[y_labels == i]
                plt.scatter(transformed_data.iloc[:, 0], transformed_data.iloc[:, 1], marker="x")
                centers = transformed_data.mean(axis=0)
                plt.scatter(
                centers[0], centers[1], s=75, marker="D", c="orange", lw=1.5, edgecolors="black"
            )

            '''centers= km.cluster_centers_
            plt.scatter(
                centers[:, 0], centers[:, 1], s=75, marker="D", c="orange", lw=1.5, edgecolors="black"
            )'''
            relative_times[method] = times_init[method] / times_init[methods[0]]

            plt.xticks(())
            plt.yticks(())
            plt.title(method, loc="left", fontsize=12)
            plt.title(
                "Time to Train %.2fx" % (relative_times[method]),
                loc="right",
                fontsize=10,
            )
        plt.suptitle("KM iterations and relative time taken to initialize Best K= "+str(best_k))
        plt.savefig("./images/"+self.dataset+"/kmeans/"+self.folder+"/best/"+"init_times")
        plt.clf()
            
    def plot_clusters_labels(self, km, k_labels, best_k):
        x_transformed = self.transformed_x
        
        centers = km.cluster_centers_ 
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
        plt.savefig("./images/"+self.dataset+"/kmeans/"+self.folder+"/best/"+"clusters_labels")
        plt.clf()
     
    def plot_clusters_pred(self, y_pred, best_k):
        _, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        
        for i, class_ in enumerate(self.classes):
            transformed_data = self.transformed_x.iloc[y_pred == i]
            ax1.scatter(transformed_data.iloc[:,0],transformed_data.iloc[:,1], cmap = "jet", edgecolor = "None", alpha=0.35, label=class_)
        ax1.set_title('KMeans Prediction')
        
        for i, class_ in enumerate(self.classes):
            y = self.y.to_numpy()
            transformed_data = self.transformed_x.iloc[y == i]
            ax2.scatter(transformed_data.iloc[:,0],transformed_data.iloc[:,1], cmap = "jet", edgecolor = "None", alpha=0.35, label=class_)
        ax2.set_title('Original Labels')
        
        plt.xlabel("Transformed Feature 1")
        plt.ylabel("Transformed Feature 2")
        plt.suptitle("Predictions Vs the Original lables clustering Best K= "+str(best_k))
        plt.legend(loc = 'best')
        plt.savefig("./images/"+self.dataset+"/kmeans/"+self.folder+"/best/"+"clusters_pred")
        plt.clf()

          
    def plot_distribution(self, y_labels, k):
        plt.title("Distribution of dataset K= "+str(k))
        plt.xlabel("Clusters")
        plt.ylabel("Count per cluster") 
        plt.hist(y_labels, bins = np.arange(k)-0.5, rwidth = 0.5)
        plt.xticks(np.arange(k))
        plt.savefig("./images/"+self.dataset+"/kmeans/"+self.folder+"/best/"+"distribution")
        plt.clf()

    def plot_silhouette(self, k, y_labels, algo_idx):
        "ref: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html"
        
        y_lower = 10        
        average = self.sil[algo_idx][k-2]
        sample = self.sample_sil[algo_idx][k-2]
        
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
        plt.savefig("./images/"+self.dataset+"/kmeans/"+self.folder+"/best/"+"sil_samples")
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

        
    