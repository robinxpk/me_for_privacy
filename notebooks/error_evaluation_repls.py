# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from ME.Data import Data
from ME.Cluster import Cluster
from ME.functions import *
from ME.ModelLists import *

# %%
data_path = r"../data/"

voe_data = pd.read_csv(f"{data_path}voe_data.csv", sep = ";", header = 0)

# %%
factor_vars = (
    # -- Survival Indicator
    "MORTSTAT",
    # -- Data set indicator
    # "SDDSRVYR",              
    # -- Exam sample weight (combined)
    "WTMEC4YR"
    # -- Follow-up time in month
    # i.e. this is "time" variable
    # "PERMTH_EXM",            
    # -- socioecnomic status level
    # "SES_LEVEL",
    # -- Age at screening
    # "RIDAGEYR",              
    # -- Is sex male
    # "male",
    # -- Area of where subject lives
    # "area",                  
    # -- Serum total thyroxine (mycro gram / dL)
    # "LBXT4",
    # -- Factor if current or past smoking
    # "current_past_smoking",  
    # -- Indicator for coronary artery disease
    # "any_cad",
    # -- Indicator that any family member has had heart attack or angina
    # "any_family_cad",        
    # -- Indicator if any cancer self report
    # "any_cancer_self_report",
    # -- BMI of subject
    # "bmi",                   
    # -- Indicator if any hyptertension
    # "any_ht",
    # -- Indicator if patient has diabetes
    # "any_diabetes",          
    # -- Categorical level of education
    # "education",
    # -- ethnicity of subject
    # "RIDRETH1",              
    # -- Ranking of physical activity
    # "physical_activity",
    # -- Indicator for heavy drinking
    # "drink_five_per_day"
    # -- Serum total cholesterol mg/dL
    # "LBXTC"         
)
#

# %%
# voe_data.loc[:, factor_vars] = voe_data.loc[:, factor_vars].astype("category")
# voe_data.loc[:, factor_vars].astype("category")
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
# %%
voe_berkson = Data(
    name = "berkson", 
    raw_data = voe_data.dropna(ignore_index = True),
    prob = 1, 
    error_type = "berkson", 
    cluster_based = True, 
    cols_excluded_from_error = ["PERMTH_EXM"]
)

# %% Initialize cox model list
# subcols for cox model
subcols = [
    # Event indicator and event time 
    "MORTSTAT", "PERMTH_EXM", 
    # Covariates
    "RIDAGEYR", "LBXT4", "bmi", 'DR1TKCAL'
    # Cannot add any of these covariables bc convergence issue due to correlation
    # 'LBXCOT',  'LBXGLU', 'LBXHGB', 'BPXPLS',"LBXTC",
]

cphs = CoxPHList(voe_true, "MORTSTAT", "PERMTH_EXM", covariables=subcols)
cphs.add_fit(voe_berkson.name, voe_berkson)


# %% initialize LM model list
lms = LMList(voe_true, "LBXT4 ~ bmi + RIDAGEYR  + DR1TKCAL")
lms.add_fit(voe_berkson.name, voe_berkson)


# %% simulation parameters
epit_sd = 2
normal_sd = 10
lognormal_sd = 1

simulation_seed = 2024
n_simulations = 100

clean_voe_data = voe_data.dropna(ignore_index = True)

# %% visualization of error effects
viz_epit = Data(
    name = "viz_epit",
    raw_data = clean_voe_data,
    prob = 1,
    error_factors = np.array([epit_sd]),
    error_type = "ePIT",
    seed = simulation_seed,
)

viz_normal = Data(
    name = "viz_normal",
    raw_data = clean_voe_data,
    prob = 1,
    error_factors = np.array([normal_sd]),
    error_type = "normal",
    seed = simulation_seed,
)

viz_lognormal = Data(
    name = "viz_lognormal",
    raw_data = clean_voe_data,
    prob = 1,
    error_factors = np.array([lognormal_sd]),
    error_type = "lognormal",
    seed = simulation_seed,
)

viz_datasets = {
    "berkson": voe_berkson,
    "ePIT": viz_epit,
    "normal": viz_normal,
    "lognormal": viz_lognormal,
}

for label, dataset in viz_datasets.items():
    print(f"KDE for BMI with and without {label} error")
    fig, ax = plt.subplots()
    dataset.raw_data["bmi"].plot.kde(ax=ax, label="Raw Data")
    dataset.masked_data["bmi"].plot.kde(ax=ax, label="Data w/Error", linestyle="--")
    ax.set_title(f'KDE w/ and w/o error [{label}]')
    ax.set_xlabel("bmi")
    ax.legend()
    fig.savefig(f"../images/viz_error_effect_bmi_{label}.png", dpi=300, bbox_inches="tight")
    plt.show()


# %%
simulation_error_configs = [
    {
        "label": "ePIT",
        "prob": 1,
        "error_factors": np.array([epit_sd]),
        "error_type": "ePIT",
    },
    {
        "label": "normal",
        "prob": 1,
        "error_factors": np.array([normal_sd]),
        "error_type": "normal",
    },
    {
        "label": "lognormal",
        "prob": 1,
        "error_factors": np.array([lognormal_sd]),
        "error_type": "lognormal",
    },
]

# %%
# simulation loop
for rep in range(n_simulations):
    current_seed = simulation_seed + rep

    for cfg in simulation_error_configs:
        sim_data = Data(
            name = f"{cfg['label']}_rep{rep + 1:03d}",
            raw_data = clean_voe_data,
            prob = cfg["prob"],
            error_factors = cfg["error_factors"],
            error_type = cfg["error_type"],
            seed = current_seed,
        )

        cphs.add_fit(sim_data.name, sim_data)
        lms.add_fit(sim_data.name, sim_data)
      
# %%
# Collect estimate tables via Fits helper methods
cox_estimates_wide = cphs.fits.table_all_estimates("\n## Cox estimates")
cox_std_err_wide = cphs.fits.table_all_estimates(
    "\n## Cox std. errors", getter=lambda mdl_fit: mdl_fit.fit.standard_errors_
)
cox_bias_wide = cphs.fits.table_all_estimates(
    "\n## Cox bias", getter=lambda mdl_fit: mdl_fit.bias
)

# %%
lm_estimates_wide = lms.fits.table_all_estimates("\n## LM estimates")
lm_std_err_wide = lms.fits.table_all_estimates(
    "\n## LM std. errors", getter=lambda mdl_fit: mdl_fit.fit.fit.bse
)
lm_bias_wide = lms.fits.table_all_estimates(
    "\n## LM bias", getter=lambda mdl_fit: mdl_fit.bias
)

# %%
def table_to_long(table, value_name):
    """
    Convert Fits.table_all_estimates output (parameter x fit) into a long DataFrame.
    """
    idx_name = table.index.name if table.index.name is not None else "index"
    long_df = table.reset_index().rename(columns={idx_name: "parameter"})
    long_df = long_df.melt(id_vars="parameter", var_name="fit_name", value_name=value_name)
    long_df["error_type"] = np.where(
        long_df["fit_name"].str.contains("_rep"),
        long_df["fit_name"].str.split("_rep").str[0],
        long_df["fit_name"],
    )
    return long_df


def add_baselines_to_long(long_df, wide_table, value_name, baseline_labels):
    """
    Ensure reference/berkson rows are present in the long table.
    """
    baseline_frames = []
    for label in baseline_labels:
        if label not in wide_table.columns:
            continue
        baseline_frames.append(
            pd.DataFrame(
                {
                    "parameter": wide_table.index,
                    "fit_name": label,
                    value_name: wide_table[label].values,
                    "error_type": label,
                }
            )
        )
    if not baseline_frames:
        return long_df
    baseline_df = pd.concat(baseline_frames, ignore_index=True)
    combined = pd.concat([long_df, baseline_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["parameter", "fit_name"], keep="last")
    return combined


# %%
cox_estimates_long = table_to_long(cox_estimates_wide, "estimate")
cox_estimates_long = add_baselines_to_long(
    cox_estimates_long, cox_estimates_wide, "estimate", ["reference", "berkson"]
)
lm_estimates_long = table_to_long(lm_estimates_wide, "estimate")
lm_estimates_long = add_baselines_to_long(
    lm_estimates_long, lm_estimates_wide, "estimate", ["reference", "berkson"]
)

parameter_name_map = {
    "Intercept": "Intercept",
    "RIDAGEYR": "age",
    "LBXT4": "serum_total_thyroxine",
    "DR1TKCAL": "total_energy_intake",
    "bmi": "bmi",
}

cox_estimates_long["parameter"] = cox_estimates_long["parameter"].map(parameter_name_map)
lm_estimates_long["parameter"] = lm_estimates_long["parameter"].map(parameter_name_map)


def add_relative_bias_to_long(long_df):
    """
    Adds a relative bias column (per row) based on the reference fit.
    """
    reference = (
        long_df[long_df["fit_name"] == "reference"][["parameter", "estimate"]]
        .rename(columns={"estimate": "reference_estimate"})
        .drop_duplicates()
    )
    merged = long_df.merge(reference, on="parameter", how="left")
    merged["reference_estimate"] = merged["reference_estimate"].replace(0, np.nan)
    merged["rel_bias"] = (
        (merged["estimate"] - merged["reference_estimate"]) / merged["reference_estimate"]
    )
    merged["rel_bias"] = merged["rel_bias"].fillna(0)
    return merged


cox_estimates_long = add_relative_bias_to_long(cox_estimates_long)
lm_estimates_long = add_relative_bias_to_long(lm_estimates_long)


# %%
cox_estimate_summary = (
    cox_estimates_long
    .groupby(["error_type", "parameter"])["estimate"]
    .agg(
        mean="mean",
        median="median",
        ci_95_lower=lambda x: x.quantile(0.025),
        ci_95_upper=lambda x: x.quantile(0.975),
    )
    .reset_index()
)
lm_estimate_summary = (
    lm_estimates_long
    .groupby(["error_type", "parameter"])["estimate"]
    .agg(
        mean="mean",
        # median="median",
        ci_95_lower=lambda x: x.quantile(0.025),
        ci_95_upper=lambda x: x.quantile(0.975),
    )
    .reset_index()
)

# %%
def add_bias_column(summary_df, long_df):
    reference_means = (
        long_df[long_df["fit_name"] == "reference"][["parameter", "estimate"]]
        .rename(columns={"estimate": "reference_mean"})
        .drop_duplicates()
    )
    merged = summary_df.merge(reference_means, on="parameter", how="left")
    merged["reference_mean"] = merged["reference_mean"].fillna(merged["mean"])
    merged["bias"] = merged["mean"] - merged["reference_mean"]
    merged["bias_rel"] = np.where(
        merged["reference_mean"] != 0,
        merged["bias"] / merged["reference_mean"],
        0,
    )
    return merged


cox_estimate_summary = add_bias_column(cox_estimate_summary, cox_estimates_long)
lm_estimate_summary = add_bias_column(lm_estimate_summary, lm_estimates_long)

cox_estimate_summary = cox_estimate_summary.sort_values(
        by="parameter",
        key=lambda col: np.where(col == "Intercept", "", col.astype(str)),
    )
lm_estimate_summary = lm_estimate_summary.sort_values(
        by="parameter",
        key=lambda col: np.where(col == "Intercept", "", col.astype(str)),
    )

# %%
# Pivot tables for quick comparison across error types
cox_bias_table = cox_estimate_summary.pivot(
    index="parameter", columns="error_type", values="bias_rel"
).round(3).sort_index(key=lambda idx: np.where(idx == "Intercept", "", idx.astype(str)))
cox_estimate_table = cox_estimate_summary.pivot(
    index="parameter", columns="error_type", values="mean"
).round(3).sort_index(key=lambda idx: np.where(idx == "Intercept", "", idx.astype(str)))

lm_bias_table = lm_estimate_summary.pivot(
    index="parameter", columns="error_type", values="bias_rel"
).round(3).sort_index(key=lambda idx: np.where(idx == "Intercept", "", idx.astype(str)))
lm_estimate_table = lm_estimate_summary.pivot(
    index="parameter", columns="error_type", values="mean"
).round(3).sort_index(key=lambda idx: np.where(idx == "Intercept", "", idx.astype(str)))

# %% print to latex
latex_str = cox_bias_table.reset_index().to_latex(index=False, float_format="%.3f")
print(latex_str)
latex_str = lm_bias_table.reset_index().to_latex(index=False, float_format="%.3f")
print(latex_str)

# %%
# Plotting helpers using relative biases
plot_error_order = ["ePIT", "normal", "lognormal", "berkson"]


def plot_rel_bias_by_parameter(long_df, title, ylim=(-15, 3)):
    box_data = long_df[long_df["error_type"].isin(plot_error_order[:-1])]
    berkson = long_df[long_df["error_type"] == "berkson"]

    g = sns.FacetGrid(box_data, col="parameter", col_wrap=2, sharey=False, height=4)
    g.map_dataframe(
        sns.boxplot,
        x="error_type",
        y="rel_bias",
        order=plot_error_order[:-1],
        palette="Set2",
    )

    # Add horizontal reference line and legend
    for ax in g.axes.flat:
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_xlabel("")


    berkson_x = plot_error_order.index("berkson")
    for ax, param in zip(g.axes.flat, g.col_names):
        subset = berkson[berkson["parameter"] == param]
        if not subset.empty:
            ax.scatter(
                berkson_x,
                subset["rel_bias"].iloc[0],
                color=sns.color_palette("Set2")[3],
                edgecolor="black",
                marker="D",
                s=40,
                label="berkson",
                zorder=5,
            )
        ax.set_xticklabels([])
    g.set_axis_labels("error type", "relative bias")
    handles = [
        plt.Line2D([0], [0], color=sns.color_palette("Set2")[i], lw=4, label=err)
        for i, err in enumerate(plot_error_order[:-1])
    ]
    handles.append(
        plt.Line2D([0], [0], marker="D", color=sns.color_palette("Set2")[3], linestyle="None", label="berkson")
    )
    g.add_legend(handles=handles, title="error type", loc="upper right")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(title)
    return g


def plot_rel_bias_aggregated(long_df, title, ylim=(-18, 3)):
    box_data = long_df[long_df["error_type"].isin(plot_error_order)]
    plt.figure(figsize=(6, 4))
    sns.boxplot(
        data=box_data,
        x="error_type",
        y="rel_bias",
        order=plot_error_order,
        palette="Set2",
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.ylim(*ylim)
    plt.xlabel("error type")
    plt.ylabel("relative bias")
    plt.title(title)
    handles = [
        plt.Line2D([0], [0], color=sns.color_palette("Set2")[i], lw=4, label=err)
        for i, err in enumerate(plot_error_order)
    ]
    plt.legend(handles=handles, title="error type", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

# %%
# Boxplots faceted by parameter
g_cox_facets = plot_rel_bias_by_parameter(cox_estimates_long, "Cox relative bias", ylim=(-4, 2.5))
g_cox_facets.fig.savefig("../images/cox_rel_bias_facets.png", dpi=300, bbox_inches="tight")

g_lm_facets = plot_rel_bias_by_parameter(lm_estimates_long, "LM relative bias", ylim=(-17, 2.5))
g_lm_facets.fig.savefig("../images/lm_rel_bias_facets.png", dpi=300, bbox_inches="tight")

# %%
# Aggregated boxplots
plt.figure()
plot_rel_bias_aggregated(cox_estimates_long, "Cox relative bias (all parameter estimates)")
plt.savefig("../images/cox_rel_bias_aggregated.png", dpi=300, bbox_inches="tight")

plt.figure()
plot_rel_bias_aggregated(lm_estimates_long, "LM relative bias (all parameter estimates)")
plt.savefig("../images/lm_rel_bias_aggregated.png", dpi=300, bbox_inches="tight")




# %% do plots with unaggregated data

# boxplot of cox estimates. of relative bias. color= error_type (berkson is only a dot), facet: estimate
# boxplot of lm estimates. of relative bias. color= error_type (berkson is only a dot), facet: estimate

# aggregated boxplot of relative bias. color = error_type  for cox
# aggregated boxplot of relative bias. color = error_type  for lm
