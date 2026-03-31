import numpy as np

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
        
        self.feature_engineered_df_cols = self.data.columns.to_list()
        
        print("\n Feature Engineering: Done")
                
        return self.data