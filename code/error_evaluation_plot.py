# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from ME.Data import Data
from ME.Cluster import Cluster
from ME.functions import *
from ME.ModelLists import *
import seaborn as sns

# %%
data_path = r"../data/"

voe_data = pd.read_csv(f"{data_path}voe_data.csv", sep = ";", header = 0)

factor_vars = (
    # -- Survival Indicator
    "MORTSTAT",
    # -- Exam sample weight (combined)
    "WTMEC4YR"
)
for col in factor_vars:
    voe_data[col] = voe_data[col].astype("category")

voe_data = voe_data.drop("WTMEC4YR", axis = 1)
# %% 
voe_true = Data(
    name = "true", 
    raw_data = voe_data.dropna(ignore_index = True),
    prob = 0, 
    error_type = "berkson", 
    cluster_based = False
)
voe_berkson = Data(
    name = "berkson", 
    raw_data = voe_data.dropna(ignore_index = True),
    prob = 1, 
    error_type = "berkson", 
    cluster_based = True, 
    cols_excluded_from_error = ["PERMTH_EXM"]
)
voe_berkson.error_evaluation
plot_df_berkson= pd.DataFrame(voe_berkson.error_evaluation).T.sort_index().assign(origin = "berkson")

# %%
# normal_sd = 10 ** 3 # NOTE: Variance should depend on scale...
records_normal = dict()
# Use KCAL variance to play around with: 
ref_sd = voe_true.raw_data["DR1TKCAL"].std()
for normal_sd_factor in np.arange(0, 1, 0.1):
    normal_sd = normal_sd_factor * ref_sd
    voe_normal= Data(
        name = f"normal_{normal_sd}", 
        raw_data = voe_data.dropna(ignore_index = True), 
        prob = 1, 
        error_factors = np.array([normal_sd]), 
        error_type="normal",
        cols_excluded_from_error = ["MORTSTAT", "WTMEC4YR"]
    )
    records_normal[normal_sd_factor] = voe_normal.error_evaluation
plot_df_normal = pd.DataFrame(records_normal).T.sort_index().assign(origin = "normal")
# %%
records_epit= dict()
for epit_sd in np.arange(0, 1, 0.1):
    voe_epit = Data(
        name = f"epit_{epit_sd}", 
        raw_data = voe_data.dropna(ignore_index = True), 
        prob = 1, 
        error_factors = np.array([epit_sd]), 
        error_type="ePIT",
        cols_excluded_from_error = ["MORTSTAT", "WTMEC4YR"]
    )
    records_epit[epit_sd] = voe_epit.error_evaluation
plot_df_epit = pd.DataFrame(records_epit).T.sort_index().assign(origin = "epit")

# %%
records_lognormal= dict()
# Use KCAL variance to play around with:
ref_sd = voe_true.raw_data["DR1TKCAL"].std()
for lognormal_sd in np.arange(0.01, 0.5, 0.1):
    voe_lognormal = Data(
        name = f"lognormal_{lognormal_sd}", 
        raw_data = voe_data.dropna(ignore_index = True), 
        prob = 1, 
        error_factors = np.array([lognormal_sd]), 
        error_type="lognormal",
        cols_excluded_from_error = ["MORTSTAT", "WTMEC4YR"]
    )
    records_lognormal[lognormal_sd] = voe_lognormal.error_evaluation
plot_df_lognormal = pd.DataFrame(records_lognormal).T.sort_index().assign(origin = "lognormal")


# %%
plot_df_list = [
    plot_df_normal, 
    plot_df_lognormal, 
    plot_df_epit, 
    plot_df_berkson
]
plot_df = pd.concat(plot_df_list).reset_index().rename(columns={"index": "error_scale"})
# %%
def plot_records(df, col): 
    scale = voe_true.raw_data[col].std()

    ax = sns.lineplot(data=df, x="error_scale", y=col, hue="origin", marker="o")
    ax.set_title("Behavior of nMSE under different error types")
    ax.set_xlabel(f"Error specific scale")
    ax.set_ylabel(f"Normalized MSE of {col}")
    ax.axhline(df.loc[df["origin"] == "berkson", col].iloc[0], linestyle = "--", color = "gray")
    plt.show()
plot_records(plot_df, "DR1TKCAL")