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
    raw_data = voe_data,
    error_type = "none"
)

voe_berkson = Data(
    name = f"berkson", 
    raw_data = voe_data.dropna(ignore_index = True), 
    seed = 1234,
    error_vars = {"DR1TKCAL": jnp.array([0.])}, 
    error_type="berkson", 
    # Exclude the error on age and bmi for now to simplify the error structure
    cluster_based=True, 
    cols_excluded_from_error = ["LBXT4", "RIDAGEYR", "bmi"]
)
# %%
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
    origin = np.unique(df.origin)[0]
    value_df = df.drop(columns="origin", errors="ignore")
    values = value_df.stack().map(
        lambda x: x.get(metric_name, np.nan)
        if isinstance(x, dict)
        else (x[metric_name] if isinstance(x, pd.Series) else x)
    )
    out = values.to_frame(name = metric_name)
    out.loc[:, "origin"] = origin
    return out.reset_index(names = ["error_scale", "iteration"])

plot_df_list = [
    extract_metric(plot_df_normal, plot_var),
    extract_metric(plot_df_lognormal, plot_var),
    extract_metric(plot_df_epit, plot_var),
    extract_metric(plot_df_berkson, plot_var),
]
plot_df = pd.concat(plot_df_list).reset_index()

corr_df_list = [
    extract_metric(plot_df_normal, "correlation"),
    extract_metric(plot_df_lognormal, "correlation"),
    extract_metric(plot_df_epit, "correlation"),
    extract_metric(plot_df_berkson, "correlation"),
]
corr_df = pd.concat(corr_df_list).reset_index()

# %%
from plotnine import ggplot, aes, geom_ribbon, geom_line, geom_point, geom_hline, geom_label, scale_y_continuous, scale_x_continuous, theme_classic, theme, element_blank, element_rect, ggsave
def run_ggplot(df, plot_var): 
    point_estimate_df = df.groupby(["origin", "error_scale"]).median().reset_index()
    upper_bound_df = df.groupby(["origin", "error_scale"]).quantile(q = 0.75).reset_index()
    lower_bound_df = df.groupby(["origin", "error_scale"]).quantile(q = 0.25).reset_index()
    upper_bound_df.rename(columns = {"DR1TKCAL": "upper"}, inplace = True)
    lower_bound_df.rename(columns = {"DR1TKCAL": "lower"}, inplace = True)
    ci_df = pd.merge(lower_bound_df, upper_bound_df, how = "left", on = ["origin", "error_scale"])

    label_df = pd.DataFrame(
        {
            "origin": ["normal", "lognormal", "ePIT"],
            "x": [0.8, 0.35, 0.75], 
            "y": [0.3, 3.5, 4.2], 
            "label": [r"$\frac{\sigma_\epsilon^2}{\widehat{var}(y)}$", r"$\sigma_{(log)}^2$", r"$\sigma_\epsilon^2$"]
        }
    )

    p = (
        ggplot(point_estimate_df, aes(x = "error_scale")) +
        geom_ribbon(data = ci_df, mapping = aes(ymin = "lower", ymax = "upper", fill = "origin"), alpha = 0.3) + 
        geom_line(mapping = aes(y = plot_var, color = "origin")) + 
        geom_point(mapping = aes(y = plot_var, color = "origin")) + 
        geom_hline(yintercept = 0.75, linetype = "--", alpha = 0.5) + 
        geom_hline(yintercept = 3, linetype = "--", alpha = 0.5) +
        geom_label(data = label_df, mapping = aes(y = "y", label = "label", x = "x")) + 
        geom_point(data = point_estimate_df.loc[point_estimate_df.origin == "berkson", :], mapping = aes(y = plot_var, color = "origin")) + 
        scale_x_continuous(
            name = "Error specific scale", 
            breaks = np.array(range(0, 11)) / 10 
        ) + 
        scale_y_continuous(
            name = r"$nMSE_{kcal}$", 
            breaks = np.array(range(0, 7)) 
        ) + 
        theme_classic(base_size = 14) + 
        theme(
            legend_position = (0.02, 0.98),
            legend_justification = (0, 1),
            legend_background = element_rect(fill = "white", color = "grey"),
            legend_title = element_blank()
        )
    )
    p.show()
run_ggplot(plot_df, plot_var = plot_var)
