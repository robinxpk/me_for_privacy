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
import jax.numpy as jnp

# %%
data_path = r"../data/"
variable_subset = ["LBXT4", "RIDAGEYR", "bmi", "DR1TKCAL"]
# Variable(s) affected by error
error_subset = ["DR1TKCAL"]
plot_var = "DR1TKCAL"
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
    cluster_based = True, 
    seed = 17,
    cols_excluded_from_error = ["LBXT4", "RIDAGEYR", "bmi"]
)

def summarize_error_run(data_obj, column_name):
    return {
        column_name: data_obj.error_evaluation[column_name],
        "correlation": data_obj.raw_data[column_name].corr(data_obj.masked_data[column_name]),
    }

plot_df_berkson = (
    pd.DataFrame({0.0: [summarize_error_run(voe_berkson, plot_var)]})
    .T
    .sort_index()
    .assign(origin = "berkson")
)

# %%
records_normal = dict()
ref_var = voe.raw_data[error_subset].var().iloc[0]
clean_data = voe_data.dropna(ignore_index = True)
# Scale Error variance using the variance of the variable: x * sigma^2
for normal_sd_factor in np.arange(0, 1.1, 0.1):
    normal_var = normal_sd_factor * ref_var
    records_normal[normal_sd_factor] = [
        summarize_error_run(
            Data(
                name = f"normal_{normal_var}", 
                raw_data = clean_data, 
                seed = 1234 + b,
                error_vars = {"DR1TKCAL": normal_var}, 
                error_type = "normal", 
                # Exclude the error on age and bmi for now to simplify the error structure
                cols_excluded_from_error = ["LBXT4", "RIDAGEYR", "bmi"]
            ),
            plot_var,
        )
        for b in range(B)
    ]

plot_df_normal = pd.DataFrame(records_normal).T.sort_index().assign(origin = "normal")
# %%
def e_sigmoid(x, b0 = -30, b1 = 4): 
    # ! Need to supply the sigmoid to the DATA object because I need this to already use THIS function for ePIT construction
    # Else, the function in the error correction is not the same as the error function. Basically.
    # MODELLING THE LOG OF THE INPUT VARIABLE!! 
    lin_mdl = b0 + b1 * jnp.log(x)
    return jax.nn.sigmoid(lin_mdl)
def e_inv_sigmoid(prob, b0 = -30, b1 = 4): 
    # Input is value(s) between 0 and 1
    # These inputs should be based on the previous e_sigmoid function 
    # !!! SEEN FROM ABOVE, to obtain x, we need to inverse the log! 
    log_odds = jnp.log(prob / (1 - prob)) 
    return jnp.exp((log_odds - b0) / b1) # log_odds = lin_mdl which we have to solve for x
records_epit= dict()
for epit_var in np.arange(0, 1, 0.1):
    records_epit[epit_var] = [
        summarize_error_run(
            Data(
                name = f"epit_{epit_var}", 
                raw_data = clean_data, 
                seed = 1234 + b,
                error_vars = {"DR1TKCAL": epit_var}, 
                error_type = "ePIT",
                cols_excluded_from_error = ["LBXT4", "RIDAGEYR", "bmi"], 
                e_sigmoid = e_sigmoid, 
                e_inv_sigmoid = e_inv_sigmoid
            ),
            plot_var,
        )
        for b in range(B)
    ]
plot_df_epit = pd.DataFrame(records_epit).T.sort_index().assign(origin = "ePIT")

# %%
records_lognormal= dict()
# Use KCAL variance to play around with:
for lognormal_var in np.arange(0.01, 0.5, 0.025):
    records_lognormal[lognormal_var] = [
        summarize_error_run(
            Data(
                name = f"lognormal_{lognormal_var}", 
                raw_data = clean_data, 
                seed = 1234 + b,
                error_vars = {"DR1TKCAL": lognormal_var}, 
                error_type = "lognormal",
                cols_excluded_from_error = ["LBXT4", "RIDAGEYR", "bmi"]
            ),
            plot_var,
        )
        for b in range(B)
    ]
plot_df_lognormal = pd.DataFrame(records_lognormal).T.sort_index().assign(origin = "lognormal")


# %%
def extract_metric(df, metric_name):
    value_df = df.drop(columns="origin", errors="ignore")
    values = value_df.stack().map(
        lambda x: x.get(metric_name, np.nan)
        if isinstance(x, dict)
        else (x[metric_name] if isinstance(x, pd.Series) else x)
    )
    out = values.groupby(level=0).mean().to_frame(name=metric_name)
    out["origin"] = df["origin"]
    return out

plot_df_list = [
    extract_metric(plot_df_normal, plot_var),
    extract_metric(plot_df_lognormal, plot_var),
    extract_metric(plot_df_epit, plot_var),
    extract_metric(plot_df_berkson, plot_var),
]
plot_df = pd.concat(plot_df_list).reset_index().rename(columns={"index": "error_scale"})

corr_df_list = [
    extract_metric(plot_df_normal, "correlation"),
    extract_metric(plot_df_lognormal, "correlation"),
    extract_metric(plot_df_epit, "correlation"),
    extract_metric(plot_df_berkson, "correlation"),
]
corr_df = pd.concat(corr_df_list).reset_index().rename(columns={"index": "error_scale"})

def format_line_label(origin, error_scale):
    if origin == "normal":
        return r"$\frac{\sigma^2_\epsilon}{\widehat{{var}}_{{kcal}}}$"
    if origin == "berkson":
        return ""
    if origin == "lognormal":
        return r"$\sigma^2_{(\log)}$"
    return rf"$\sigma^2_\epsilon$"


def plot_records(df, col, xmax, title, ylabel): 
    origins = df["origin"].drop_duplicates().tolist()
    palette_colors = sns.color_palette(n_colors=len(origins))
    palette = dict(zip(origins, palette_colors))

    ax = sns.lineplot(data=df, x="error_scale", y=col, hue="origin", marker="o", palette=palette)
    ax.set_title(title)
    ax.set_xlabel(f"Error specific scale")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.05, xmax)
    if "berkson" in df["origin"].values:
        ax.axhline(df.loc[df["origin"] == "berkson", col].iloc[0], linestyle = "--", color = "gray")
    # ax.axhline(df.loc[df["origin"] == "ePIT", col].iloc[7], linestyle = "--", color = "gray")
    if ax.legend_ is not None:
        ax.legend_.set_title(None)
    line_endpoints = (
        df.sort_values("error_scale")
        .groupby("origin", as_index=False)
        .tail(1)
    )
    for _, row in line_endpoints.iterrows():
        label = format_line_label(row["origin"], row["error_scale"])
        if not label:
            continue
        ax.annotate(
            label,
            (row["error_scale"], row[col]),
            xytext=(-10, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": palette[row["origin"]],
                "linewidth": 1.25,
            }
        )
    plt.show()
plot_records(
    plot_df,
    plot_var,
    xmax = 1,
    title = "Behavior of nMSE under different error types",
    ylabel = f"Normalized MSE of {plot_var}",
)
plot_records(
    corr_df,
    "correlation",
    xmax = 4,
    title = "Behavior of correlation under different error types",
    ylabel = f"Correlation between original and error-touched {plot_var}",
)
print(voe_berkson.evaluate_errors())
# %%
best_berkson_seed = None
best_berkson_score = -np.inf

for seed in range(9999, 10 ** 6):
    current_berkson = Data(
        name = "berkson",
        raw_data = voe_data.dropna(ignore_index = True),
        error_type = "berkson",
        cluster_based = True,
        seed = seed,
        cols_excluded_from_error = ["LBXT4", "RIDAGEYR", "bmi"]
    )
    current_score = current_berkson.evaluate_errors()["DR1TKCAL"]
    if current_score > best_berkson_score:
        best_berkson_seed = seed
        best_berkson_score = current_score

print(best_berkson_seed)

# %%
