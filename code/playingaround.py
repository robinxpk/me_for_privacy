# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


# %%
data_path = r"data/"

nhanes_demo = pd.read_sas(f"{data_path}DEMO_L.xpt")
nhanes_diet1 = pd.read_sas(f"{data_path}DR1IFF_L.xpt")


key = "SEQN"

# make sure key has the same dtype
nhanes_demo[key] = nhanes_demo[key].astype("int64")
nhanes_diet1[key] = nhanes_diet1[key].astype("int64")

# join side-by-side (each food item gets the person’s demographics)
nhanes = pd.merge(
    nhanes_demo, nhanes_diet1, on=key, how="inner", validate="one_to_many"
)

# nhanes = pd.concat([nhanes_demo, nhanes_diet1], ignore_index=True, sort=False)


# %%
# Set up automatic layout
num_cols = len(nhanes.columns)
ncols = 3  # number of plots per row
nrows = (num_cols + ncols - 1) // ncols  # rows needed

plt.figure(figsize=(15, 4 * nrows))

for i, col in enumerate(nhanes.columns, 1):
    plt.subplot(nrows, ncols, i)

    if pd.api.types.is_numeric_dtype(nhanes[col]):
        nhanes[col].dropna().plot(kind="hist", bins=30, edgecolor="black")
        plt.title(f"{col} (Histogram)")
    else:
        nhanes[col].value_counts(dropna=False).plot(kind="bar")
        plt.title(f"{col} (Bar Plot)")

    plt.tight_layout()

plt.show()


# %%

nhanes_2 = nhanes.loc[:, ["RIDAGEYR", "DR1IKCAL"]]
nhanes_2 = nhanes_2.dropna()

# %%
# visualize data ####
# TODO: high dimensions??
# Visualize the dataset
plt.figure(figsize=(10, 6))
nhanes_2.plot.scatter(x="RIDAGEYR", y="DR1IKCAL", figsize=(10, 6))
plt.xlabel("Age")
plt.ylabel("Energy in kcal")
plt.show()


# %%
# choose epsilon ####
# Function to plot k-distance graph
def plot_k_distance_graph(x, k):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(x)
    distances, _ = neigh.kneighbors(x)
    distances = np.sort(distances[:, k - 1])
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel("Points")
    plt.ylabel(f"{k}-th nearest neighbor distance")
    plt.title("K-distance Graph")
    plt.ylim(0, 25)
    plt.axhline(y=2, color="red", linestyle="--", linewidth=1.5, label="y = 2")
    plt.axhline(y=3, color="blue", linestyle="--", linewidth=1.5, label="y = 3")
    plt.show()


# Plot k-distance graph

minpts = nhanes_2.shape[1] * 2
k_start = minpts - 1
plot_k_distance_graph(nhanes_2, k=k_start)

# %%

# TODO: choose epsilon where we see the ellbow
epsilon = 5

# do DBSCAN  ####
dbscan = DBSCAN(
    eps=epsilon,
    min_samples=minpts,
    metric="euclidean",  # TODO: choose appropriate metric
    p=2,  # TODO: choose power of Minkowski distance (thigher p -> large distances have more influence (bad?)
)
clusters = dbscan.fit_predict(nhanes_2)
# %%
# visualize results (again, how for high dims?)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    nhanes_2.iloc[:, 0],
    nhanes_2.iloc[:, 1],
    c=clusters,
    cmap="viridis",
    s=10,
    alpha=0.5,
)
plt.colorbar(scatter)
plt.title("DBSCAN Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# %%
# Print number of clusters and noise points
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# TODO: how to get all points in a cluster?
# TODO: color noise points vs clusters to see where they are -> noise gets different error
# clusters.labels_()  # cluster labels for each point in the data set

# %%
