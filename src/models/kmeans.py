from sklearn.cluster import KMeans
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

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

class KMeansClustering_:
    def __init__(self, k=None, feature_names=None):
        self.k = k
        self.feature_names = feature_names
        self.model = KMeans(n_clusters=self.k, n_init=10)
        self.labels = None
        self.dataset = X_dataframe
        
    def fit(self, dataset):
        self.model.fit(dataset)
        self.labels = self.model.labels_
        
    def predict(self, dataset):
        return self.model.predict(dataset)
    
    def evaluate(self):
        kmeans_evaluator = Evaluator(dataset=self.dataset, labels=self.labels)
        evaluation = kmeans_evaluator.evaluate()
        return evaluation
    
pipeline = KMeansClustering_(k=3, feature_names=feature_columns)
# pipeline.fit(dataset=X_dataframe)
# result = pipeline.evaluate()
# print(result)