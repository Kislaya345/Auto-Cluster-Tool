from sklearn.mixture import GaussianMixture
import numpy as np

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from src.evaluation.metrics import Evaluator
from src.visualization.plots import _plot_2d

class gmm_pipeline:
    def __init__(self, data):
        self.dataset = data
        self.model_name = "Gaussian Mixture Model (GMM)"
        self.n_components = self._find_optimal_components()
        self.model = GaussianMixture(n_components=self.n_components, random_state=43)
        self.labels = None
        
    def _find_optimal_components(self):
        # BIC score - lower is better
        # BIC penalizes complexity so it won't just keep adding clusters
        best_bic = np.inf
        best_k = 2
        
        for k in range(2, 8):
            gmm = GaussianMixture(n_components=k, random_state=43, covariance_type="tied")
            gmm.fit(self.dataset)
            bic = gmm.bic(self.dataset)
            aic = gmm.aic(self.dataset)
            # print(f"k={k} | BIC={bic:.2f} | AIC={aic:.2f}")
            
            if bic < best_bic:
                best_bic = bic
                best_k = k
        
        print(f"Optimal components (k): {best_k}")
        print(f"Optimal BIC (lowest BIC found for the dataset): {best_bic}")
        return best_k
    
    def fit_predict(self):
        self.labels = self.model.fit_predict(self.dataset)
        
        min_cluster_size = 10
        from collections import Counter
        counts = Counter(self.labels)
    
        for label, count in counts.items():
            if count < min_cluster_size:
                # Mark these points as -1 (noise) or reassign
                self.labels[self.labels == label] = -1
    
        return self.labels
    
    def evaluate(self):
        if self.labels is None:
            print("Run fit_predict first")
            return None
        evaluator = Evaluator(self.dataset, self.labels)
        evaluation = evaluator.evaluate()
        return evaluation
    
    def plot(self):
        _plot_2d(
            data=self.dataset,
            labels=self.labels,
            name="GMM",
            n_components=2
        )