# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 01:21:45 2024

@author: Fedaa Almomani
         Shefaa Mestarihi
         Saja Zreqat
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# Read the data
data = pd.read_excel("C:\\Users\\user\\Downloads\\customer_dataset.xlsx")
null_values = data.isnull().sum()
print(data.info())
# Encoding categorical variables using LabelEncoder
encoder = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = encoder.fit_transform(data[col])

# Select a portion of the data (features you want to use)
#selected_features = ['Region', 'Payment_mode', 'how_they_buy', 'Amount_in_usd', 'Product_type', 'Time Of Day']
selected_features = [ 'Time Of Day', 'Amount_in_usd']


data_selected = data[selected_features]

# Prepare the data for clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_selected)

# Step 2: Choose a Clustering Algorithm (K-means)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(data_scaled)

# Step 3: Evaluate Silhouette Score
silhouette_avg = silhouette_score(data_scaled, kmeans.labels_)
print("Silhouette Score:", silhouette_avg)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# رسم المجموعات
plt.figure(figsize=(10, 8))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')
plt.title('Customer Clusters (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()