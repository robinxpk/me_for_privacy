# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ME.Data import Data
from ME.Cluster import Cluster
from ME.functions import *
from ME.ModelLists import *
import seaborn as sns
import jax

# %%
data_path = r"../data/"
variable_subset = ["LBXT4", "RIDAGEYR", "bmi", "DR1TKCAL"]
# Variable(s) affected by error
error_subset = ["DR1TKCAL"]
# Points making up the later error-boxplot
B = 20

voe_data = pd.read_csv(f"{data_path}voe_data.csv", sep = ";", header = 0)

factor_vars = (
    # -- Survival Indicator
    "MORTSTAT",
    # -- Exam sample weight (combined)
    "WTMEC4YR"
)
for col in factor_vars:
    voe_data[col] = voe_data[col].astype("category")

voe_data = voe_data.loc[:, variable_subset].dropna(ignore_index = True)

# %% 
voe = Data(
    name = "true", 
    raw_data = voe_data.dropna(ignore_index = True),
    error_type = "none"
)
voe_berkson = Data(
    name = "berkson", 
    raw_data = voe_data.dropna(ignore_index = True),
    error_type = "berkson", 
    cluster_based = True
)
voe_berkson.error_evaluation
plot_df_berkson= pd.DataFrame(voe_berkson.error_evaluation).T.sort_index().assign(origin = "berkson")

# %%
records_normal = dict()
ref_var = voe.raw_data[error_subset].var()
clean_data = voe_data.dropna(ignore_index = True)
# Scale Error variance using the variance of the variable: x * sigma^2
for normal_sd_factor in np.arange(0, 1, 0.1):
    normal_var = normal_sd_factor * ref_var
    records_normal[normal_sd_factor] = [
        Data(
            name = f"normal_{normal_var.values[0]}", 
            raw_data = clean_data, 
            seed = 1234 + b,
            error_vars = {"DR1TKCAL": normal_var}, 
            error_type = "normal", 
            # Exclude the error on age and bmi for now to simplify the error structure
            cols_excluded_from_error = ["LBXT4", "RIDAGEYR", "bmi"]
        ).error_evaluation[error_subset]
        for b in range(B)
    ]

plot_df_normal = pd.DataFrame(records_normal).T.sort_index().assign(origin = "normal")
# %%
records_epit= dict()
for epit_var in np.arange(0, 1, 0.1):
    records_epit[epit_var] = [
        Data(
            name = f"epit_{epit_var}", 
            raw_data = clean_data, 
            seed = 1234 + b,
            error_vars = {"DR1TKCAL": epit_var}, 
            error_type = "ePIT",
            cols_excluded_from_error = ["LBXT4", "RIDAGEYR", "bmi"]
        ).error_evaluation[error_subset]
        for b in range(B)
    ]
plot_df_epit = pd.DataFrame(records_epit).T.sort_index().assign(origin = "epit")

# %%
records_lognormal= dict()
# Use KCAL variance to play around with:
for lognormal_var in np.arange(0.01, 0.5, 0.05):
    records_lognormal[lognormal_var] = [
        Data(
            name = f"lognormal_{lognormal_var}", 
            raw_data = clean_data, 
            seed = 1234 + b,
            error_vars = {"DR1TKCAL": lognormal_var}, 
            error_type = "lognormal",
            cols_excluded_from_error = ["LBXT4", "RIDAGEYR", "bmi"]
        ).error_evaluation[error_subset]
        for b in range(B)
    ]
plot_df_lognormal = pd.DataFrame(records_lognormal).T.sort_index().assign(origin = "lognormal")


# %%
plot_var = "DR1TKCAL"

def extract_plot_var(df, plot_var):
    value_df = df.drop(columns="origin", errors="ignore")
    values = value_df.stack(dropna=False).map(
        lambda x: x.get(plot_var, np.nan)
        if isinstance(x, dict)
        else (x[plot_var] if isinstance(x, pd.Series) else x)
    )
    out = values.groupby(level=0).mean().to_frame(name=plot_var)
    out["origin"] = df["origin"]
    return out


plot_df_list = [
    extract_plot_var(plot_df_normal, plot_var),
    extract_plot_var(plot_df_lognormal, plot_var),
    extract_plot_var(plot_df_epit, plot_var),
    extract_plot_var(plot_df_berkson, plot_var),
]
plot_df = pd.concat(plot_df_list).reset_index().rename(columns={"index": "error_scale"})
# %%
def plot_records(df, col): 
    scale = voe.raw_data[col].std()

    ax = sns.lineplot(data=df, x="error_scale", y=col, hue="origin", marker="o")
    ax.set_title("Behavior of nMSE under different error types")
    ax.set_xlabel(f"Error specific scale")
    ax.set_ylabel(f"Normalized MSE of {col}")
    ax.axhline(df.loc[df["origin"] == "berkson", col].iloc[0], linestyle = "--", color = "gray")
    plt.show()
plot_records(plot_df, plot_var)
# %%
