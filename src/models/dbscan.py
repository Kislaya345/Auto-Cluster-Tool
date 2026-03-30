from sklearn.cluster import DBSCAN

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from src.evaluation.metrics import Evaluator 
from src.visualization.plots import _plot_2d

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
    
    def plots(self):
        _plot_2d(
            data=self.dataset,
            labels=self.labels,
            name="DBSCAN",
            n_components=2
        )