from sklearn.cluster import AgglomerativeClustering

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from src.evaluation.metrics import Evaluator
from data.load import Load_Data
from data.processed_data import Preprocess
from src.features.feature_engineering import Feature_Engineer

loader = Load_Data()
dataset = loader.load_data(path='C:/Users/kisla/Downloads/archive/wine_dataset.csv')
raw_data = loader.dataset
processor = Preprocess(dataset=raw_data, path=loader.path)
processor.preprocess()
feature_engine = Feature_Engineer(data=processor.X_dataframe, dataframe_cols=processor.feature_columns)
X_dataframe = feature_engine.perform()
feature_columns = feature_engine.feature_engineered_df_cols

class AgglomerativeClustering_:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.metric = 'euclidean'
        self.linkage = 'ward'
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage, metric=self.metric)
    
    def fit_predict(self, dataset):
        self.labels = self.model.fit_predict(dataset)
        return self.labels
    
    def evaluate(self):
        evaluator = Evaluator(dataset=X_dataframe, labels=self.labels)
        evaluation = evaluator.evaluate()
        return evaluation

# pipeline = AgglomerativeClustering_(n_clusters=3)
# pipeline.fit_predict(dataset=X_dataframe)
# result = pipeline.evaluate()
# print(result)