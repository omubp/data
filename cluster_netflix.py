import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Load dataset
df = pd.read_csv("netflix_titless.csv", encoding='latin1')

# 2. Select important columns
df = df[['release_year', 'duration_minutes']]

# 3. Clean duration (extract number)
df['duration_minutes'] = df['duration_minutes'].str.extract('(\d+)')
df['duration_minutes'] = pd.to_numeric(df['duration_minutes'], errors='coerce')

# 4. Handle missing values
df = df.dropna()

# 5. Define features
X = df[['release_year', 'duration_minutes']]

# 6. Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# 7. Show clustered data
print("\nClustered Data:\n", df.head())

# 8. Cluster centers
print("\nCluster Centers:\n", kmeans.cluster_centers_)

# 9. Visualization
plt.scatter(X['release_year'], X['duration_minutes'], c=df['Cluster'])
plt.xlabel("Release Year")
plt.ylabel("Duration")
plt.title("Netflix Content Clustering")
plt.show()
