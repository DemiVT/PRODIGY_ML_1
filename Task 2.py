import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('customer_data.csv')  # Ensure this file path is correct

# Features for clustering
X = data[['total_purchases', 'average_purchase_value', 'frequency']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust n_clusters as needed
data['cluster'] = kmeans.fit_predict(X_scaled)

print(data.head())
