import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler

# 1. Get the directory of the current file (src/features)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up TWO levels to reach the project root (AUTO CLUSTER TOOL)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from data.load import Load_Data
from data.processed_data import Preprocess

loader = Load_Data()
dataset = loader.load_data(path='C:/Users/kisla/Downloads/archive/wine_dataset.csv')
raw_data = loader.dataset
processor = Preprocess(dataset=raw_data, path=loader.path)
processor.preprocess()
dataframe_cols = processor.feature_columns
processed_df = processor.processed_df

class Feature_Engineer:
    def __init__(self, data, dataframe_cols):
        self.data = data.copy()
        self.dataframe_cols = dataframe_cols
        self.feature_engineered_df_cols = None
        
    def perform(self):
        
        # Fixing skewness in the data
        for column in self.dataframe_cols:
            skew_val = self.data[column].skew()
            if abs(skew_val) > 1.0:
                print("The skew value is: ", skew_val)
                print(f"Applying Log Transform to {column} (Skew: {skew_val:.2f})")
                # We use log1p (log of 1+x) to avoid issues with zero values
                self.data[column] = np.log1p(self.data[column])    
        else: 
            print("No skewness in the data")
                
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            
        for col in numeric_cols:
            scaler = StandardScaler()
            self.data[col] = scaler.fit_transform(self.data[[col]])
        
        self.feature_engineered_df_cols = self.data.columns.to_list()
                
        return self.data