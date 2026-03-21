import csv
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

correct_root = r'C:\Users\kisla\OneDrive\Desktop\Everything\Auto Cluster Tool'

if correct_root not in sys.path:
    sys.path.insert(0, correct_root)
    
from load import Load_Data

load_datast = Load_Data()
load_datast.load_data(path='C:/Users/kisla/Downloads/archive/wine_dataset.csv')
dataset = load_datast.dataset
path = load_datast.path

class Preprocess:
    def __init__(self, dataset, path):
        self.dataset = dataset
        self.path = path
        self.columns = None
        
    def preprocess(self):
        is_valid = False
        with open(self.path, 'r') as f:
            sample = f.read(2048) # Read small sample
            try:
                dialect = csv.Sniffer().sniff(sample) # Check if it has consistent delimiter
                has_header = csv.Sniffer().has_header(sample)
                print(f"Valid CSV with delimiter '{dialect.delimiter}'. Header present: {has_header}")
                is_valid = True
            except csv.Error: 
                print("Not a valid csv format.")
                return False
            
        # Getting the target column in the dataset
        categorical_cols = dataset.select_dtypes(exclude=['float']).columns # Get the categoric columns
        if len(categorical_cols) > 0:
            target_col = dataset[categorical_cols].nunique().idxmin()
            y = dataset[target_col]
        
        else: 
            target_col = dataset.nunique().idxmin()
            y = dataset[target_col]
            
        # Flling missing values
        self.columns = self.dataset.columns.to_list()
        dataframe = pd.DataFrame(dataset, columns=self.columns)
        
        for column in self.columns:
            if pd.api.types.is_numeric_dtype(dataframe[column]):
                # calculate missing values
                na_vals_count = dataframe[column].isna().sum()
                
                if na_vals_count > 0:
                    dataframe[column].fillna(dataframe[column].mean(), inplace=True)
            
            if pd.api.types.is_bool_dtype(dataframe[column]):
                dataframe[column] = dataframe[column].astype(int)
            
            if pd.api.types.is_object_dtype(dataframe[column]):
                total_rows = len(dataframe.shape[0])
                unique_val_str = dataframe[column].nunique()
                
                if unique_val_str > total_rows * 0.9 or unique_val_str <= 1:
                    print(f"Dropping {column}: Not a useful category.")
                    dataframe.drop(columns=[column], inplace=True)
            
            else: pass
                
            dataframe = pd.get_dummies(dataframe, drop_first=True)
            
            return dataframe
        
        # Scaling features
        scaler = StandardScaler()
        scaled_dataframe_array = scaler.fit_transform(dataframe)
        scaled_dataframe = pd.DataFrame(scaled_dataframe_array, columns=self.columns)
        
        return scaled_dataframe
        
preprocess = Preprocess(dataset=dataset, path=path)
result = preprocess.preprocess()
print(result)
