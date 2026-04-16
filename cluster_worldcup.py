import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Load dataset (Sheet3)
df = pd.read_excel("world_cup_results.xlsx", sheet_name="WorldCups")

# 2. Clean Attendance column
df['Attendance'] = df['Attendance'].astype(str)
df['Attendance'] = df['Attendance'].str.replace('.', '', regex=False)
df['Attendance'] = pd.to_numeric(df['Attendance'], errors='coerce')

# 3. Select features (only numeric)
X = df[['GoalsScored', 'MatchesPlayed', 'Attendance']]

# 4. Handle missing values
X = X.fillna(X.mean())

# 5. Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# 6. Show clustered data
print("\nClustered Data:\n")
print(df[['Year', 'GoalsScored', 'MatchesPlayed', 'Attendance', 'Cluster']])

# 7. Show cluster centers
print("\nCluster Centers:\n")
print(kmeans.cluster_centers_)

# 8. Visualization
plt.scatter(X['GoalsScored'], X['Attendance'], c=df['Cluster'])
plt.xlabel("GoalsScored")
plt.ylabel("Attendance")
plt.title("World Cup Clustering")
plt.show()
