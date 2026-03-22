import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if root_path not in sys.path:
#     sys.path.insert(0, root_path)

# 1. Get the directory of the current file (src/features)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up TWO levels to reach the project root (AUTO CLUSTER TOOL)
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.load import Load_Data
from data.processed_data import Preprocess
from src.visualization.plots import feature_dist_scatter_plot, feature_correlation_plot, _plot_2d, _plot_3d, _scree_plot

data_loader = Load_Data()
dataframe = data_loader.load_data(path='C:/Users/kisla/Downloads/archive/wine_dataset.csv')
processor = Preprocess(dataset=dataframe, path=data_loader.path)
processed_dataframe = processor.preprocess()
processed_dataframe_cols = processor.feature_columns

for col in processed_dataframe_cols:
    print(f"\nDataset Median for {col} is {processed_dataframe[col].median()}")
    print(f"Dataset describe for {col} is \n{processed_dataframe[col].describe()}")

# Feature Distribution Plot

feature_dist_scatter_plot(dataframe=processed_dataframe, feature_names=processor.feature_columns, bins=20)

# Correlation plot

feature_correlation_plot(processor.X_dataframe.corr(), annot=True)

# PCA EDA

class PCA_Plot:
    def __init__(self, processed_dataframe, n_components):
        self.processed_dataframe = processed_dataframe
        self.n_components = n_components
        
    def fit(self):
        pca_viz = PCA(n_components=self.n_components)
        data_pca = pca_viz.fit_transform(self.processed_dataframe)
        
        n_selected = pca_viz.n_components_
        pc_names = [f'PC{i+1}' for i in range(n_selected)]
        
        loadings = pd.DataFrame(
            pca_viz.components_.T,
            columns=pc_names,
            index=processor.feature_columns
        )
        
        top_drivers = {pc: loadings[pc].abs().idxmax() for pc in pc_names}
        for pc, driver in top_drivers.items():
            print(f"Top Driver for {pc}: {driver}")
            
        if n_selected == 2:

            _plot_2d(data_pca, pca_viz, top_drivers, processor)
        elif n_selected >= 3: 

            _plot_3d(data_pca[:, :3], n_selected, top_drivers, processor)

        _scree_plot(pca_viz.explained_variance_ratio_, n_selected)