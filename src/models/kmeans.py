import os
import sys
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from src.evaluation.metrics import Evaluator

class KMeansClustering_:
    def __init__(self, k=None, dataset=None):
        self.k = k
        self.model = KMeans(n_clusters=self.k, n_init=10)
        self.labels = None
        self.dataset = dataset
        self.n_components = 2
        
    def fit(self):
        self.model.fit(self.dataset)
        self.labels = self.model.labels_
        return self.labels
        
    def predict(self, dataset):
        return self.model.predict(dataset)
    
    def evaluate(self):
        kmeans_evaluator = Evaluator(dataset=self.dataset, labels=self.labels)
        evaluation = kmeans_evaluator.evaluate()
        return evaluation
    
    def plot(self):
        pca = PCA(n_components=self.n_components)
        X_pca = pca.fit_transform(self.dataset)
        
        plt.figure(figsize=(10, 6))
        for label in set(self.labels):
            mask = self.labels == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       label=f'Cluster {label}', alpha=0.8, edgecolors='k')
        
        plt.title(f"GMM Clusters (n_components={self.n_components})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()