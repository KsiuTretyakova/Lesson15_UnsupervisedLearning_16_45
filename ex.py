import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    'Середня кількість покупок на місяць': [10, 5, 20, 15, 8, 3, 50, 45, 7, 2],
    'Середній чек': [100, 50, 200, 150, 80, 30, 500, 400, 450, 70],
    'Частота відвідувань': [30, 10, 5, 7, 90, 40, 20, 5, 50, 25]
}

df = pd.DataFrame(data)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Кластер'] = kmeans.fit_predict(df)

centroids = kmeans.cluster_centers_


plt.scatter(df['Середня кількість покупок на місяць'], df['Середній чек'], c=df['Кластер'], cmap='viridis', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, label='Centroids', marker='x')
plt.title("Result clusters")
plt.xlabel("Середня кількість покупок на місяць")
plt.ylabel("Середній чек")
plt.colorbar(label='Кластер')
plt.show()
