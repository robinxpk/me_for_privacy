import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

class Cluster(): 
    def __init__(self, fit, data):
        self.fit = fit
        self.data = data
        self.names = data.columns
        self.labels = self.fit.labels_
        self.centers = self.fit.cluster_centers_
        # Inertia: Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.
        self.inertia = self.fit.inertia_
        self.unique_clusters = np.unique(self.labels)
    
    def evaluate(self): 
        # Data as an np.array to use np.array-like calculations
        X = np.asarray(self.data)

        # Squared distance for each observation to the assigned cluster center
        sqd_dists = np.sum((X - self.centers[self.labels]) ** 2, axis = 1)

        # list of arrays, one entry per cluster
        data_per_cluster = [sqd_dists[self.labels == c] for c in self.unique_clusters]

        fig, ax = plt.subplots(
            figsize=(max(12, len(self.unique_clusters) * 0.02), 6)  # wider for many clusters
        )

        ax.boxplot(data_per_cluster, showfliers=True)

        ax.set_xlabel("cluster")
        ax.set_ylabel("squared distance")

        # x-tick positions are 1..K for K clusters
        ax.set_xticks(range(1, len(self.unique_clusters) + 1))
        ax.set_xticklabels(self.unique_clusters, rotation=90, fontsize=4)  # shrink font

        step = 10
        ticks = np.arange(1, len(self.unique_clusters) + 1, step)
        tick_labels = self.unique_clusters[::step]

        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)

        fig.tight_layout()
        plt.show()


        df = pd.DataFrame({"cluster": self.labels, "sq_dist": sqd_dists})

        # per-cluster quantiles
        g = df.groupby("cluster")["sq_dist"]
        q25 = g.quantile(0.25)   # first quartile :contentReference[oaicite:0]{index=0}
        q75 = g.quantile(0.75)   # third quartile

        iqr = q75 - q25          # interquartile range

        # clusters sorted by IQR (ascending; use .sort_values(ascending=False) for descending)
        order = iqr.sort_values().index

        # data list in this order, one array per cluster
        data_per_cluster = [df.loc[df["cluster"] == c, "sq_dist"].values for c in order]

        fig, ax = plt.subplots(figsize=(16, 4))
        ax.boxplot(data_per_cluster, showfliers=True)  # one box per entry :contentReference[oaicite:1]{index=1}

        ax.set_ylabel("squared distance")

        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=90, fontsize=4)
        ax.set_xlabel("cluster (sorted by IQR)")

        plt.tight_layout()
        plt.show()
