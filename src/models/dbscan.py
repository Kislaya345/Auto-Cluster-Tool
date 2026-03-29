from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from src.evaluation.metrics import Evaluator 

class DBSCAN_pipeline:
    def __init__(self, dataset=None):
        self.dataset = dataset
    
    def fit_predict(self):
        dbscan = DBSCAN()
        dbscan.fit_predict(self.dataset)
        self.labels = dbscan.labels_
    
    def evaluate(self):
         dbscan_evaluator = Evaluator(dataset=self.dataset, labels=self.labels)
         evaluation = dbscan_evaluator.evaluate()
         return evaluation
    
    def plot(self):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.dataset)
        
        plt.figure(figsize=(10, 6))
        for label in set(self.labels):
            mask = self.labels == label
            color = 'black' if label == -1 else None  # noise points for DBSCAN
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=f'Cluster {label}' if label != -1 else 'Noise',
                    alpha=0.3 if label == -1 else 0.8,
                    c=color, edgecolors='k')
        
        plt.title()
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()