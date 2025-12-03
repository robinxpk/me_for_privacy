# %%
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from ME.Data import Data

def plot_k_distance_graph(x, k):
    # choose epsilon for DBScan 
    # Function to plot k-distance graph
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
    # plt.axhline(y=2, color="red", linestyle="--", linewidth=1.5, label="y = 2")
    # plt.axhline(y=3, color="blue", linestyle="--", linewidth=1.5, label="y = 3")
    plt.show()

# %%
data_path = r"./data/"

nhanes_demo = pd.read_sas(f"{data_path}DEMO_L.xpt")
nhanes_diet1 = pd.read_sas(f"{data_path}DR1IFF_L.xpt")
nhanes_body_measures = pd.read_sas(f"{data_path}BMX_L.xpt")
key = "SEQN"

# make sure key has the same dtype
nhanes_demo[key] = nhanes_demo[key].astype("int64")
nhanes_diet1[key] = nhanes_diet1[key].astype("int64")
nhanes_body_measures[key] = nhanes_body_measures[key].astype("int64")

# join side-by-side (each food item gets the person’s demographics)
nhanes_raw = pd.merge(
    nhanes_demo, nhanes_diet1, on=key, how="inner", validate="one_to_many"
)

nhanes_raw = pd.merge(
    nhanes_raw, nhanes_body_measures, on=key, how="inner", validate="many_to_many"
)

# %%
nhanes_raw = nhanes_raw.loc[:, [
        "DR1DAY", # string: Intake day of the week
        # 1: Sunday, 2: Monday, ..., 7: Saturday, .: Missing
        "DR1LANG", # string: Language respondent used mostly
        # 1: English, 2: Spanish, 3: English and Spanish, 4: Other, .: Missing
        "RIDAGEYR", # int: Age
        "DR1IKCAL", # float: Energy (kcal)
        "DR1IPROT", # float: Protein (gm)
        "DR1ICARB", # float: Carbs (gm) 
        "DR1ISUGR", # float: Sugar (gm)
        "DR1IFIBE", # float: Fibre (gm)
        "DR1ITFAT", # float: Fat (gm)
        "DR1ICHOL", # float: Cholesterol (mg)
        "BMXHT" # float: height (cm)
    ]
]
nhanes_raw = nhanes_raw.dropna(ignore_index=True) 

lookup_dict = {
    "DR1DAY": {
        1: "sunday", 
        2: "monday", 
        3: "tuesday",
        4: "wednesday", 
        5: "thursday", 
        6: "friday", 
        7: "saturday"
    },
    "DR1LANG": {
        1: "english", 
        2: "spanish", 
        3: "english and spanish", 
        4: "other"
    }
}

# %%
nhanes_raw.DR1DAY = [lookup_dict["DR1DAY"][number] for  number in nhanes_raw.DR1DAY]
nhanes_raw.DR1LANG = [lookup_dict["DR1LANG"][number] for  number in nhanes_raw.DR1LANG]

nhanes_raw.DR1DAY = nhanes_raw.DR1DAY.astype("category")
nhanes_raw.DR1LANG = nhanes_raw.DR1LANG.astype("category")

# %%
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
nhanes_epit = Data(
    raw_data = nhanes_raw.dropna(ignore_index = True), 
    prob = 0.5, 
    error_factors = np.array([2]), 
    error_type="ePIT"
)

nhanes_normal= Data(
    raw_data = nhanes_raw.dropna(ignore_index = True), 
    prob = 0.5, 
    error_factors = np.array([50]), 
    error_type="normal"
)

nhanes_lognormal = Data(
    raw_data = nhanes_raw.dropna(ignore_index = True), 
    prob = 0.5, 
    error_factors = np.array([0.3]), 
    error_type="lognormal"
)

# %% 
# Visualize effect of error
nhanes_epit.viz_error_effect("BMXHT")
nhanes_normal.viz_error_effect("BMXHT")
nhanes_lognormal.viz_error_effect("BMXHT")

# nhanes_epit.viz_error_effect("DR1LANG")
# nhanes_lognormal.viz_error_effect("DR1LANG")
# nhanes_normal.viz_error_effect("DR1LANG")

# %% 
# Visualize ePIT 
nhanes_epit.ePIT_viz("BMXHT")


# %%
# Run Clustering
nhanes = Data(
    raw_data = nhanes_raw.dropna(ignore_index = True), 
    prob = 0.5, 
    error_factors = np.array([0.5]), 
    error_type="ePIT"
)
# %%
# Scatter plot of variables in data
# ! Run only if not too many variables contained in df
pd.plotting.scatter_matrix(nhanes.raw_data, alpha = 0.2, s = 5)
pd.plotting.scatter_matrix(nhanes.masked_data, alpha = 0.2, s = 5)

# %%
nhanes.raw_data.DR1DAY.value_counts().plot(kind = "bar")
plt.show()
nhanes.masked_data.DR1DAY.value_counts().plot(kind = "bar")
plt.show()
nhanes.raw_data.DR1LANG.value_counts().plot(kind = "bar")
plt.show()
nhanes.masked_data.DR1LANG.value_counts().plot(kind = "bar")
plt.show()

# %%
# Plot k-distance graph
# For now, we do not use DBScan on categorical data as our distance measure only works on numericals
nhanes.drop_column("DR1DAY")
nhanes.drop_column("DR1LANG")

# %%
distance_measure = "euclidean"

minpts = nhanes.raw_data.shape[1] * 2
k_start = minpts - 1
plot_k_distance_graph(nhanes.raw_data, k=k_start)
epsilon = 1
# do DBSCAN  ####
dbscan = DBSCAN(
    eps=epsilon,
    min_samples=minpts,
    metric=distance_measure,  # TODO: choose appropriate metric
    p=2,  # TODO: choose power of Minkowski distance (thigher p -> large distances have more influence (bad?)
)
clusters_raw = dbscan.fit_predict(nhanes.raw_data)
# Print number of clusters and noise points
n_clusters = len(set(clusters_raw)) - (1 if -1 in clusters_raw else 0)
n_noise = list(clusters_raw).count(-1)
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
nhanes.raw_data["cluster"] = clusters_raw

minpts = nhanes.masked_data.shape[1] * 2
k_start = minpts - 1
plot_k_distance_graph(nhanes.masked_data, k=k_start)
epsilon = 1
# do DBSCAN  ####
dbscan = DBSCAN(
    eps=epsilon,
    min_samples=minpts,
    metric=distance_measure,  # TODO: choose appropriate metric
    p=2,  # TODO: choose power of Minkowski distance (thigher p -> large distances have more influence (bad?)
)
clusters_masked = dbscan.fit_predict(nhanes.masked_data)
n_clusters = len(set(clusters_masked)) - (1 if -1 in clusters_masked else 0)
n_noise = list(clusters_masked).count(-1)
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
nhanes.masked_data["cluster"] = clusters_masked

# %%
# visuaraw_lize results (again, how for high dims?)
plotted_y = "DR1ITFAT"
plotted_x = "DR1ICHOL"
cluster_id = 1


plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    nhanes.raw_data[nhanes.raw_data.cluster == cluster_id].loc[:, plotted_y],
    nhanes.raw_data[nhanes.raw_data.cluster == cluster_id].loc[:, plotted_x],
    c=clusters_raw[nhanes.raw_data.cluster == cluster_id],
    cmap="viridis",
    s=10,
    alpha=0.3,
)
plt.colorbar(scatter)
plt.title("DBSCAN Clustering Results - Raw Data")
plt.xlabel(plotted_x)
plt.ylabel(plotted_y)
plt.show()

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    nhanes.masked_data[nhanes.raw_data.cluster == cluster_id].loc[:, [plotted_y]],
    nhanes.masked_data[nhanes.raw_data.cluster == cluster_id].loc[:, [plotted_x]],
    c=clusters_masked[nhanes.raw_data.cluster ==  cluster_id],
    cmap="viridis",
    s=10,
    alpha=0.3,
)
plt.colorbar(scatter)
plt.title("DBSCAN Clustering Results - Added Error")
plt.xlabel(plotted_x)
plt.ylabel(plotted_y)
plt.show()


# %%
### Visuals 

# Ratio of data (All vs Ratio p)
# -> Mention equivalence of uncertainty expressed in variance for numericals
# Suggestions: error on all vars bc then no "true value" in data (ePIT more complex; could error into PIT there)

# Additive Error
# + Easy Bayes 
# + Nice Cluster interpretation
# - Selective handling of values (e.g. non-negatives) and Outliers (e.g. higher additive error in outliers cases)
# x_new = x + error

# Multiplicative Error
# Handles Outliers
# Easy Bayes
# x_new = x * exp(error)

# ePIT Error
# + No issues regarding Outliers
# + No issues with values
# - Bayes? 
# x_ij= f(x_ij|x_j) 

# Error on categorical data: 
# Suggest: Random draws of emprical distribution to not allow inference due to prior knowledge e.g. based on paper content (--> Here, error ratio p vs all error is relvant) 

# num: DR1IPROT
# cat: DR1LANG

# OPTIONAL: Cluster results dependent on error




