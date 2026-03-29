from sklearn.cluster import KMeans
import os
import sys

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