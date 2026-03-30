import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.visualization.plots import feature_dist_scatter_plot, feature_correlation_plot, _plot_2d, _plot_3d, _scree_plot

class EDA:
    def __init__(self, X_dataframe=None, feature_names=None):
        self.X_dataframe = X_dataframe
        self.feature_names = feature_names
        
    def run(self):
        for col in self.feature_names:
            print(f"\nDataset Median for {col} is {self.X_dataframe[col].median()}")
            print(f"Dataset description for {col} is \n{self.X_dataframe[col].describe()}")

        # Feature Distribution Plot
        feature_dist_scatter_plot(dataframe=self.X_dataframe, feature_names=self.feature_names, bins=20)

        # Correlation plot
        feature_correlation_plot(self.X_dataframe.corr(), annot=True)
        
        print("\n Exploratory Data Analysis: Done")