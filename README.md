# digit-clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
import pandas as pd
from google.colab import files
uploaded = files.upload()

data_competition= pd.read_csv('data.csv')
data_competition.head()
ID=data_competition['ID']
data_competition= data_competition.drop(['ID'], axis=1)
data_competition= preprocessing.normalize(data_competition, axis=0)
type(data_competition)

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=60, random_state=0)
data_reduced = tsne.fit_transform(data_scaled)
# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95, random_state=0)  # Choose the number of components as needed
reduced_data = pca.fit_transform(data_scaled)

# Apply K-Means Clustering on the reduced-dimensional data
kmeans = KMeans(n_clusters=12)  # Change the number of clusters as needed

labels= kmeans.fit_predict(reduced_data)

# Now you have your cluster labels in 'cluster_labels'
# You can further analyze your clusters or visualize them as needed
import matplotlib.pyplot as plt

# Plot the clusters
plt.figure(figsize=(10, 6))

# Plot each cluster
for cluster in range(kmeans.n_clusters):
    cluster_data = reduced_data[labels == cluster]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}')

# Plot the cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='black', label='Cluster Centers')

plt.title('K-Means Clustering after PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
from sklearn.manifold import TSNE

# Fit the t-SNE model to your normalized data

tsne_data = tsne.fit_transform(data_reduced)  # Assuming you've already normalized your data

# Convert the t-SNE transformed data to a DataFrame
tsne_df = pd.DataFrame(tsne_data, columns=['t-SNE Component 1', 't-SNE Component 2'])

# Plot the t-SNE results
plt.figure(figsize=(10, 8))
plt.scatter(tsne_df['t-SNE Component 1'], tsne_df['t-SNE Component 2'], alpha=0.5)
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
# Assuming data_reduced is your reduced-dimensional data from PCA

# Apply k-means clustering
#k = 12 # Example: number of clusters
#kmeans_tsne = KMeans(n_clusters=k,random_state=42)
kmeans_tsne =KMeans(init='k-means++', n_clusters=12, n_init=60)
kmeans_tsne.fit(data_reduced)

# Predict the cluster labels for each data point
'''********************************'''
labels = kmeans_tsne.predict(data_reduced)

# Get the centroids
centroids_tsne = kmeans_tsne.cluster_centers_

# Print the centroids
print("Centroids:")
print(centroids_tsne)

# Print the cluster labels for each data point
print("Cluster Labels:")
print(labels)
import matplotlib.pyplot as plt

# Plot the reduced data points
plt.figure(figsize=(8, 6))
plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=kmeans_tsne.labels_, cmap='viridis', s=10, alpha=0.5)
plt.scatter(centroids_tsne[:, 0], centroids_tsne[:, 1], marker='x', c='red', s=100, label='Centroids')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('t-SNE & K-Means Clustering')
plt.legend()
plt.grid(True)
plt.show()
#gave me a lower score!!from yellowbrick.cluster import KElbowVisualizer

# Assuming data_competition is your dataset with 784 columns

# Instantiate the KMeans model with explicit n_init
kmeans_tsne = KMeans(n_init=10)

# Create the elbow visualizer
visualizer_tsne = KElbowVisualizer(kmeans_tsne, k=(1, 14))

# Fit the visualizer to the data
visualizer_tsne.fit(data_competition)

# Show the plot
visualizer_tsne.show()
# add
from sklearn.mixture import GaussianMixture

# Assuming `data_reduced` is your t-SNE reduced data
# Determine the optimal number of components using BIC
n_components_range = np.arange(1, 13)

bic = []
lowest_bic = np.infty
best_gmm = None

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gmm.fit(data_reduced)
    bic_score = gmm.bic(data_reduced)
    bic.append(bic_score)
    if bic_score < lowest_bic:
        lowest_bic = bic_score
        best_gmm = gmm

# Plot the BIC scores
plt.figure(figsize=(8, 6))
plt.plot(n_components_range, bic, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('BIC')
plt.title('BIC Scores for Different Number of Components')
plt.show()

# Use the best GMM to predict the clusters
'''*********************************'''
#labels = best_gmm.predict(data_reduced)
labels = best_gmm.predict(data_reduced)


# Print the cluster labels for each data point
print("Cluster Labels:")
print(labels)
from sklearn.decomposition import IncrementalPCA
import numpy as np

# Generate example data (replace this with your data)
n_samples, n_features = 1000, 785
X = np.random.randn(n_samples, n_features)

# Initialize IPCA with the desired number of components
n_components = 3
ipca = IncrementalPCA(n_components=n_components)

# Process batches of data (e.g., streaming or mini-batch processing)
batch_size = 100
for i in range(0, len(X), batch_size):
    X_batch = X[i:i+batch_size]
    ipca.partial_fit(X_batch)

# Retrieve the principal components
principal_components = ipca.components_

# Transform the data using the learned principal components
X_transformed = ipca.transform(X)

# Print the transformed data
print("Transformed Data:")
print(X_transformed)
import pandas as pd

# Assuming data_competition is your NumPy array

# Convert the NumPy array to a DataFrame
df_competition = pd.DataFrame(data_competition)

# Specify the file path for saving the Excel fileimport pandas as pd

# Assuming data_competition is your NumPy array and labels is your cluster labels array

# Convert the NumPy array to a DataFrame
df_competition = pd.DataFrame(data_competition)

# Create a DataFrame for the labels
df_labels = pd.DataFrame({'label': labels})

# Concatenate the "ID" column with the labels DataFrame
_df = pd.concat([ID, df_labels], axis=1)

# Save the DataFrame to a CSV file
_df.to_csv('sample_submission.csv', index=False)
