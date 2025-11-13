# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from ME.Data import Data

# %%
data_path = r"../data/"

nhanes_demo = pd.read_sas(f"{data_path}DEMO_L.xpt")
nhanes_diet1 = pd.read_sas(f"{data_path}DR1IFF_L.xpt")

key = "SEQN"

# make sure key has the same dtype
nhanes_demo[key] = nhanes_demo[key].astype("int64")
nhanes_diet1[key] = nhanes_diet1[key].astype("int64")

# join side-by-side (each food item gets the person’s demographics)
nhanes_raw = pd.merge(
    nhanes_demo, nhanes_diet1, on=key, how="inner", validate="one_to_many"
)


# %%

nhanes_raw = nhanes_raw.loc[:, [
        # "DR1DAY", # Intake day of the week
        # "DR1LANG", # Language respondent used mostly
        # "RIDAGEYR", # Age
        "DR1IKCAL", # Energy (kcal)
        "DR1IPROT", # Protein (gm)
        "DR1ICARB", # Carbs (gm) 
        "DR1ISUGR", # Sugar (gm)
        "DR1IFIBE", # Fibre (gm)
        "DR1ITFAT", # Fat (gm)
        "DR1ICHOL" # Cholesterol (mg)
    ]
]

# 1) Boolean pro variable ziehen (50%?)
# 2) If TRUE: Fehler 
#     Hierzu: 
#     - Numerisch: ePIT auf Variable und Variable in Normalverteilung transformieren
#         Normalen Fehlerterm aufaddieren: 
#         Fehlervarianz randomly ziehen
#         Grund für random draw: Kein gesondertes Behandeln der Outlier
#             sonder es kann durch eine sehr hohe Fehlervarianz auch zu Outliern kommen. So erhalten Outlier nicht nur möglicherweise einen Fehlerterm, sondern bestehende Outlier werden durch mögliche neue Outlier "gemasked". 
#             Natürlich kann es auch dazu kommen, dass Outlier noch weiter in die Outlier-Richtung gezogen werden. Hier machen wir uns aber erstmal keine Sorgen, weil das durch gutes Masking (hoffentlich) weniger relevant ist.
#     - Kategorisch: Zufälliges Ziehen aus...
#         ... möglichen Kategorien (Uniform) --> Enthält keine Informationen bzgl der empirischen Verteilung der Kategorien
#         ... empirische Verteilung der Kategorien --> Möglicherweise kann so die empirische Verteilung beibehalten werden, allerdings weiß ich nicht, ob das das Signal nicht weird macht? Für uniform Fehler scheint es mir einfacher zu korrigieren
#         Nicht sicher was davon besser wäre?
# 3) Re-Fit DBScan and display same clusters

# %%
nhanes = Data(raw_data = nhanes_raw.dropna(ignore_index = True), prob = 0.5, error_factors = np.array([0.5]))

# %%
# Scatter plot of variables in data
# ! Run only if not too many variables contained in df
# pd.plotting.scatter_matrix(nhanes.raw_data, alpha = 0.2, s = 5)
# pd.plotting.scatter_matrix(nhanes.masked_data, alpha = 0.2, s = 5)

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

minpts = nhanes.raw_data.shape[1] * 2
k_start = minpts - 1
plot_k_distance_graph(nhanes.raw_data, k=k_start)
epsilon = 3

# do DBSCAN  ####
dbscan = DBSCAN(
    eps=epsilon,
    min_samples=minpts,
    metric="euclidean",  # TODO: choose appropriate metric
    p=2,  # TODO: choose power of Minkowski distance (thigher p -> large distances have more influence (bad?)
)
clusters_raw = dbscan.fit_predict(nhanes.raw_data)
# Print number of clusters and noise points
n_clusters = len(set(clusters_raw)) - (1 if -1 in clusters_raw else 0)
n_noise = list(clusters_raw).count(-1)
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")


minpts = nhanes.masked_data.shape[1] * 2
k_start = minpts - 1
plot_k_distance_graph(nhanes.masked_data, k=k_start)
epsilon = 5

# do DBSCAN  ####
dbscan = DBSCAN(
    eps=epsilon,
    min_samples=minpts,
    metric="euclidean",  # TODO: choose appropriate metric
    p=2,  # TODO: choose power of Minkowski distance (thigher p -> large distances have more influence (bad?)
)
clusters_masked = dbscan.fit_predict(nhanes.masked_data)
n_clusters = len(set(clusters_masked)) - (1 if -1 in clusters_masked else 0)
n_noise = list(clusters_masked).count(-1)
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# %%
# # visualize results (again, how for high dims?)
# plt.figure(figsize=(10, 6))
# scatter = plt.scatter(
#     nhanes.raw_data.iloc[:, 0],
#     nhanes.raw_data.iloc[:, 1],
#     c=clusters_raw,
#     cmap="viridis",
#     s=10,
#     alpha=0.5,
# )
# plt.colorbar(scatter)
# plt.title("DBSCAN Clustering Results")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()

# %%

# TODO: how to get all points in a cluster?
# TODO: color noise points vs clusters to see where they are -> noise gets different error
# clusters.labels_()  # cluster labels for each point in the data set

# %%
labels = np.asarray(clusters_raw)
K = np.unique(labels).size
# cmap = plt.get_cmap("tab10", K)  
nhanes.raw_data.insert(0, "labels", labels)
# pd.plotting.scatter_matrix(nhanes.raw_data, alpha = 0.2, s = 5, c = labels)


labels = np.asarray(clusters_masked)
K = np.unique(labels).size
# cmap = plt.get_cmap("tab10", K)  
nhanes.masked_data.insert(0, "labels", labels)
# pd.plotting.scatter_matrix(nhanes.masked_data, alpha = 0.2, s = 5, c = labels)
# # %%
# # Grab a few points to plot how their behavior / their cluster changed based on added error
# def idxquantile(s, q=0.5, *args, **kwargs):
#     qv = s.quantile(q, *args, **kwargs)
#     return (s.sort_values()[::-1] <= qv).idxmax()

# reference_col = "DR1IKCAL"
# plot_idcs = np.array([idxquantile(nhanes.raw_data.loc[:, reference_col], q = p) for p in np.array(range(0, 100, 10)) / 100])

# raw_plot_labels = nhanes.raw_data.loc[plot_idcs, :]["labels"].unique()

# masked_plot_labels = nhanes.masked_data.loc[plot_idcs, :]["labels"].unique()


# # Plot only certain labels in output
# nhanes.raw_data.color = np.where(nhanes.raw_data["labels"] == 0, "green", "red")
# pd.plotting.scatter_matrix( 
#     nhanes.raw_data.loc[[label in raw_plot_labels for label in nhanes.raw_data.labels], :],
#     alpha = 0.2, s = 5, c = nhanes.raw_data.loc[[label in raw_plot_labels for label in nhanes.raw_data.labels], "color"]
# )

# pd.plotting.scatter_matrix( 
#     nhanes.masked_data.loc[[label in masked_plot_labels for label in nhanes.masked_data.labels], :],
#     alpha = 0.2, s = 5, c = np.where(nhanes.raw_data["labels"] == 0, "green", "red")
# )


# %%
# Grab a few points to plot how their behavior / their cluster changed based on added error
# def idxquantile(s, q=0.5, *args, **kwargs):
#     qv = s.quantile(q, *args, **kwargs)
#     return (s.sort_values()[::-1] <= qv).idxmax()

# reference_col = "DR1IKCAL"
# plot_idcs = np.array([
#     idxquantile(nhanes.raw_data.loc[:, reference_col], q=p)
#     for p in np.array(range(0, 100, 10)) / 100
# ])

# raw_plot_labels = nhanes.raw_data.loc[plot_idcs, "labels"].unique()
# masked_plot_labels = nhanes.masked_data.loc[plot_idcs, "labels"].unique()

# ---- RAW DATA PLOT ----
# Subset once
# raw_mask = nhanes.raw_data["labels"].isin(raw_plot_labels)
# raw_df   = nhanes.raw_data.loc[raw_mask, :]
# raw_df = nhanes.raw_data
# unique_labels = np.unique(nhanes.raw_data.labels)
# n_clusters = len(unique_labels[unique_labels != -1])


sns.scatterplot(
    data=nhanes.raw_data,
    x="DR1IKCAL",
    y="DR1IPROT",
    hue="labels",        # like color = labels in R
    palette="Set1",      # distinct colors for each label
)

# Colormap: one color per cluster, black for noise
# colors = plt.cm.get_cmap("tab20", len(unique_labels))
# for k in unique_labels:
#     class_mask = labels == k

#     if k == -1:
#         # noise
#         col = "k"
#         marker = "x"
#         lab = "noise"
#     else:
#         col = colors(k)
#         marker = "o"
#         lab = f"cluster {k}"

#     plt.scatter(
#         X[class_mask, 0],
#         X[class_mask, 1],
#         c=[col],
#         s=10,
#         marker=marker,
#         label=lab,
#         alpha=0.6,
#     )

# plt.legend(markerscale=2)
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.title(f"DBSCAN clusters (k={n_clusters})")
# plt.show()


# # Colors for *exactly* these rows
# # raw_colors = np.where(raw_df["labels"] == 0, "green", "red")

# pd.plotting.scatter_matrix(
#     raw_df,
#     alpha=0.2,
#     s=5,
#     c=raw_colors
# )

# # ---- MASKED DATA PLOT ----
# masked_mask = nhanes.masked_data["labels"].isin(masked_plot_labels)
# masked_df   = nhanes.masked_data.loc[masked_mask, :]

# # Colors for *exactly* these rows (do NOT use nhanes.raw_data here)
# masked_colors = np.where(masked_df["labels"] == 0, "green", "red")

# pd.plotting.scatter_matrix(
#     masked_df,
#     alpha=0.2,
#     s=5,
#     c=masked_colors
# )

# %%
