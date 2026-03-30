import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

def feature_dist_scatter_plot(dataframe, feature_names, bins):
    
    # Number of rows to be kept 
    number_of_features = len(feature_names)
    
    cols = math.ceil(math.sqrt(number_of_features))
    rows = math.ceil(number_of_features / cols)
     
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = axes.flatten()

    for i, feature in enumerate(feature_names):
        axes[i].hist(dataframe[feature], bins=bins, edgecolor='black')
        axes[i].set_title(f"Distribution of {feature}")
        
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()
    
def feature_correlation_plot(correlation_matrix, annot):
    sns.heatmap(correlation_matrix, annot=annot, cmap='coolwarm')
    plt.show()
    
def dendogram(data):
    linked = linkage(data, method='ward')
        
    plt.figure(figsize=(12, 6))
    dendrogram(linked)
    plt.title("Agglomerative Clustering Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.show()

def _plot_2d(data, labels, name, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(data)
    
    plt.figure(figsize=(8, 6))
    
    
    if name == "DBSCAN":
        for label in set(labels):
            mask = labels == label
            color = 'black' if label == -1 else None  # noise points for DBSCAN
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=f'Cluster {label}' if label != -1 else 'Noise',
                    alpha=0.3 if label == -1 else 0.8,
                    c=color, edgecolors='k')
    
    for label in set(labels):
        mask = labels == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=f'Cluster {label}', alpha=0.8, edgecolors='k')
    plt.title(f"{name} Clusters (n_components={n_components})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_comparision(scores: dict):
    valid = {k: v for k, v in scores.items() if v is not None}
    
    models = list(valid.keys())
    
    silhouette = [valid[m]['Silhouette_Score'] for m in models]
    dav_b = [valid[m]['Davies_Bouldin_Score'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, axes = plt.subplots(figsize=(10,5))
    
    bars1 = axes.bar(x=x-width/2, height=silhouette, width=width, label='Silhouette (higher=better)', color='steelblue')
    bars2 = axes.bar(x=x+width/2, height=dav_b, width=width, label='Davies-Bouldin (lower=better)', color='coral')
    
    for bar in bars1:
        axes.text(bar.get_x() + bar.get_width() /2, bar.get_height() + 0.01, f'{bar.get_height():.2f}', ha='center', fontsize=9)
        
    for bar in bars2:
        axes.text(bar.get_x() + bar.get_width() /2, bar.get_height() + 0.01, f'{bar.get_height():.2f}', ha='center', fontsize=9)
        
    axes.set_xticks(x)
    axes.set_xticklabels(models)
    axes.set_title("Model comparison")
    axes.legend()
    plt.tight_layout()
    plt.show()