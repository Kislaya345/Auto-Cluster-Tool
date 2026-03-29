from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class Evaluator: 
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels
        self.result = None
        
    def evaluate(self):
        unique_labels = set(self.labels) - {-1}  # excluding noise
    
        if len(unique_labels) < 2:
            print(f"Skipping evaluation — only {len(unique_labels)} cluster found")
            return None
        
        self.result = {}
        # Adding silhouette_score to dictionary
        self.result['Silhouette_Score'] = silhouette_score(self.dataset, self.labels)
        # Adding davies_bouldin_score to dictionary
        self.result['Davies_Bouldin_Score'] = davies_bouldin_score(self.dataset, self.labels)
        # Adding calinski_harabasz_score to dictionary
        self.result['Calinski_Harabasz_Score'] = calinski_harabasz_score(self.dataset, self.labels)
        
        return self.result