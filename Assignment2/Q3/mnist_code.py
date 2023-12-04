import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_data()

    def load_data(self):
        with open(self.file_path, 'rb') as file_stream:
            loaded_data = pickle.load(file_stream)
        self.features = loaded_data['X']
        self.labels = loaded_data['Y']

class DimensionalityReducer:
    def __init__(self, method='PCA'):
        self.method = method

    def reduce_dimensions(self, features, n_components=2):
        if self.method == 'PCA':
            reducer = PCA(n_components=n_components)
        elif self.method == 't-SNE':
            reducer = TSNE(n_components=n_components, random_state=42)
        else:
            raise ValueError("Invalid dimensionality reduction method.")
        return reducer.fit_transform(features)

class ClusterAnalyzer:
    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def apply_kmeans(self, data):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        return kmeans.fit_predict(data)

class DataVisualizer:
    def __init__(self):
        self.figure, self.axes = plt.subplots(1, 2, figsize=(12, 6))

    def scatter_plot(self, data, kmeans_result, title, ax):
        ax.scatter(data[:, 0], data[:, 1], c=kmeans_result, cmap='tab10')
        ax.set_title(title)

    def display_plots(self):
        plt.show()

# Main script
data_loader = DataLoader('data/mnist_small.pkl')

pca_reducer = DimensionalityReducer(method='PCA')
pca_result = pca_reducer.reduce_dimensions(data_loader.features)

kmeans1_analyzer = ClusterAnalyzer(n_clusters=10, random_state=42)
kmeans_result1 = kmeans1_analyzer.apply_kmeans(pca_result)

tsne_reducer = DimensionalityReducer(method='t-SNE')
tsne_result = tsne_reducer.reduce_dimensions(data_loader.features)

kmeans2_analyzer = ClusterAnalyzer(n_clusters=10, random_state=42)
kmeans_result2 = kmeans2_analyzer.apply_kmeans(tsne_result)

visualizer = DataVisualizer()

visualizer.scatter_plot(pca_result, kmeans_result1, 'PCA Visualization with K-Means Clustering', visualizer.axes[0])
visualizer.scatter_plot(tsne_result, kmeans_result2, 't-SNE Visualization with K-Means Clustering', visualizer.axes[1])

visualizer.display_plots()
