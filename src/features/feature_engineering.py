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
                
        # fixing correlation
        corr = self.data.corr(method='pearson').abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        # Finding pair with highest correlation
        high_corr_pairs = [column for column in upper.columns if any(upper[column] > 0.85)]
            
        # if high_corr_pairs: 
        #     print(f"Merging high correlated features: {high_corr_pairs}")
        #     self.data['Correlated_feature_mean'] = self.data[high_corr_pairs].mean(axis=1)
        #     self.data.drop(columns=high_corr_pairs[1::2], inplace=True)    
            
        # else: 
            # print("No significant multi column found")
            
        # Distribution Flagging and Feature Transform
        clump_threshold=0.5, 
        bimodal_kurt_threshold=-0.8
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            
        for col in numeric_cols:
        #     # Identifying clumping (variance > threshold)
        #     top_val_freq = self.data[col].value_counts(normalize=True).iloc[0]
        #     if top_val_freq > clump_threshold:
        #             mode_val = self.data[col].mode()[0]
        #             # Create Flag: 1 if in the clump, 0 otherwise
        #             self.data[f'{col}_is_clump'] = (self.data[col] == mode_val).astype(float)*0.1
        #             # print(f"Flagged Clumping in '{col}' at value {mode_val}")

        #     # Identifying bimodality (Negative Kurtosis indicates two peaks/flatness)
        #     if self.data[col].kurt() < bimodal_kurt_threshold:
        #         # Create a Threshold Flag based on the median
        #         median_val = self.data[col].median()
        #         self.data[f'{col}_is_high_peak'] = (self.data[col] > median_val).astype(float)*0.1
        #         # print(f"Flagged Bimodality in '{col}'")

            # Transform and Replace
            # Use Yeo-Johnson to 'smooth' the distribution for K-Means
            scaler = StandardScaler()
            self.data[col] = scaler.fit_transform(self.data[[col]])
        
        self.feature_engineered_df_cols = self.data.columns.to_list()
                
        return self.data