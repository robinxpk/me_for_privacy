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

# %%
class CoxPHList: 
    def __init__(
            self, 
            # Data sets 
            true_data, 
            # variables
            duration_col, event_col, covariables = []
        ):
        self.data = true_data
        self.duration_col = duration_col
        self.event_col = event_col
        self.covariables = self.data.raw_data.columns
        if len(covariables) > 0: 
            self.covariables = covariables
        
        self.cph_ref = CoxPHFitter().fit(self.data.raw_data[self.covariables], duration_col = self.duration_col, event_col = self.event_col)

        self.fits = Fits()
        self.add_fit("reference", self.data)

    
    def add_fit(self, name, data):
        self.fits.add_new_fit(
            Fit(
                name, 
                data,
                CoxPHFitter().fit(data.masked_data[self.covariables], duration_col = self.duration_col, event_col = self.event_col), 
                self.cph_ref
            )
        )

class FitList: 
    # TODO: All fits should be inherited from this object to not be redundant.... 
    def __init__(self): 
        pass
# %%
class LMList():
    def __init__(self, true_data, formula):
        # Check how super works. Do not remember right now...
        # super.__init__(self)
        self.data = true_data
        self.formula = formula
        self.lm_ref = LM(data = self.data.raw_data, formula = self.formula)

        self.fits = Fits()
        self.add_fit("reference", self.data)

    def add_fit(self, name, data):
        self.fits.add_new_fit(
            Fit(
                name, 
                data,
                LM(data.masked_data, formula = self.formula), 
                self.lm_ref
            )
        )

# %%
class LM: 
    # Wrapper to make lm-fit quack like a duck (cph-fit)
    def __init__(self, data, formula): 
            self.data = data
            self.formula = formula
            self.fit = self._fit(self.data, self.formula)
            self.params_ = self.fit.params

    def _fit(self, data, form = "LBXT4 ~ RIDAGEYR + bmi + LBXTC"): 
        return smf.ols(form, data=data).fit()


class Fits: 
    def __init__(self):
        self.fits = {}

    def __getattr__(self, name): 
        # called only if normal attribute lookup fails
        try:
            return self.fits[name]
        except KeyError:
            raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def add_new_fit(self, Fit): 
        self.fits[Fit.name]= Fit
    
    def table_all_estimates(
            self, table_name, table_as_markdown = False,
            # Allow to define a getter function to be able to extract different summary statistics: 
            getter = lambda mdl_fit: mdl_fit.fit.params_
            # or e.g.: 
            # getter=lambda mdl_fit: mdl_fit.fit.standard_errors_
            # getter=lambda mdl_fit: mdl_fit.fit.summary["p"]
        ): 
        print(table_name)
        table = pd.concat(
            [df.rename(name) for name, df in zip(self.fits.keys(), [getter(mdl_fit) for mdl_fit in self.fits.values()])],
            axis=1,
        )

        if table_as_markdown: 
            return table.to_markdown()
        return table

    def boxplot_all_estimates(
        self, plot_name, y_lab = "Estimate", marker_in_boxplot = "reference", dot_color = "r",
        # Allow to define a getter function to be able to extract different summary statistics: 
        getter = lambda mdl_fit: mdl_fit.fit.params_
        # or e.g.: 
        # getter=lambda mdl_fit: mdl_fit.fit.standard_errors_
        # getter=lambda mdl_fit: mdl_fit.fit.summary["p"]
    ): 
        table = pd.concat(
            [df.rename(name) for name, df in zip(self.fits.keys(), [getter(mdl_fit) for mdl_fit in self.fits.values()])],
            axis=1,
        )
        plt_table = table.transpose()
        
        fig, ax = plt.subplots(figsize=(12, 4))
        plt_table.boxplot(ax = ax)
        ax.scatter(np.arange(1, len(table) + 1), plt_table.loc[marker_in_boxplot, :], color=dot_color, zorder=3)

        plt.title(plot_name)
        plt.ylabel(y_lab)
        ax.tick_params(axis = "x", rotation=90)
        plt.tight_layout()
        plt.show()

class Fit: 
    def __init__(self, name, data, fitted_mdl, ref_mdl):
        self.name = name
        self.data = data
        self.fit = fitted_mdl
        self.ref_fit = ref_mdl
        self.bias = self.fit.params_ - self.ref_fit.params_


# %%
data_path = r"../data/"

voe_data = pd.read_csv(f"{data_path}voe_data.csv", sep = ";", header = 0)

# %%
factor_vars = (
    # -- Survival Indicator
    "MORTSTAT",
    # -- Data set indicator
    "SDDSRVYR",              
    # -- Exam sample weight (combined)
    "WTMEC4YR",
    # -- Follow-up time in month
    # i.e. this is "time" variable
    # "PERMTH_EXM",            
    # -- socioecnomic status level
    "SES_LEVEL",
    # -- Age at screening
    # "RIDAGEYR",              
    # -- Is sex male
    "male",
    # -- Area of where subject lives
    "area",                  
    # -- Serum total thyroxine (mycro gram / dL)
    # "LBXT4",
    # -- Factor if current or past smoking
    "current_past_smoking",  
    # -- Indicator for coronary artery disease
    "any_cad",
    # -- Indicator that any family member has had heart attack or angina
    "any_family_cad",        
    # -- Indicator if any cancer self report
    "any_cancer_self_report",
    # -- BMI of subject
    # "bmi",                   
    # -- Indicator if any hyptertension
    "any_ht",
    # -- Indicator if patient has diabetes
    "any_diabetes",          
    # -- Categorical level of education
    "education",
    # -- ethnicity of subject
    "RIDRETH1",              
    # -- Ranking of physical activity
    "physical_activity",
    # -- Indicator for heavy drinking
    "drink_five_per_day"
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
# %%
voe_berkson.viz_error_effect("LBXT4")
# voe_berkson.prior_cluster.evaluate()
# voe_berkson.post_cluster.evaluate()


# %%50
epit_sd = 2
voe_epit = Data(
    name = f"epit_{epit_sd}", 
    raw_data = voe_data.dropna(ignore_index = True), 
    prob = 1, 
    error_factors = np.array([epit_sd]), 
    error_type="ePIT"
)
# %%
voe_epit.ePIT_viz("LBXT4")
voe_epit.viz_error_effect("LBXT4")
# voe_epit.prior_cluster.evaluate()
# voe_epit.post_cluster.evaluate()

# %%
normal_sd = 10
voe_normal= Data(
    name = f"normal_{normal_sd}", 
    raw_data = voe_data.dropna(ignore_index = True), 
    prob = 1, 
    error_factors = np.array([normal_sd]), 
    error_type="normal"
)
# %%
voe_normal.viz_error_effect("LBXT4")
# voe_normal.prior_cluster.evaluate()
# voe_normal.post_cluster.evaluate()

# %%
lognormal_sd = 1
voe_lognormal = Data(
    name = f"lognormal_{lognormal_sd}", 
    raw_data = voe_data.dropna(ignore_index = True), 
    prob = 1, 
    error_factors = np.array([lognormal_sd]), 
    error_type="lognormal"
)
# %%
voe_lognormal.viz_error_effect("LBXT4")
# voe_lognormal.prior_cluster.evaluate()
# voe_lognormal.post_cluster.evaluate()


# %% Fit models
cphs = CoxPHList(voe_true, "MORTSTAT", "PERMTH_EXM")
cphs.add_fit(voe_berkson.name, voe_berkson)
cphs.add_fit(voe_epit.name, voe_epit)
cphs.add_fit(voe_normal.name, voe_normal)
cphs.add_fit(voe_lognormal.name, voe_lognormal)
# %%
subcols = ["MORTSTAT", "PERMTH_EXM", "RIDAGEYR", "LBXT4", "bmi", "LBXTC"]
cphs_small = CoxPHList(voe_true, "MORTSTAT", "PERMTH_EXM", covariables = subcols)
cphs_small.add_fit(voe_berkson.name, voe_berkson)
cphs_small.add_fit(voe_epit.name, voe_epit)
cphs_small.add_fit(voe_normal.name, voe_normal)
cphs_small.add_fit(voe_lognormal.name, voe_lognormal)

# %% 
print(cphs.fits.table_all_estimates(table_name = "\n## Estimates", table_as_markdown = True))
# ! marker_in_boxplot uses name specified in table
cphs.fits.boxplot_all_estimates(plot_name = "Estimates", marker_in_boxplot="reference")
# %%
print(cphs.fits.table_all_estimates(table_name = "\n## Std. Error", table_as_markdown=True, getter = lambda mdl_fit: mdl_fit.fit.standard_errors_))
cphs.fits.boxplot_all_estimates(plot_name = "Std. Error", getter = lambda mdl_fit: mdl_fit.fit.standard_errors_)
# %%
print(cphs.fits.table_all_estimates(table_name = "\n## p-values", table_as_markdown=True, getter = lambda mdl_fit: mdl_fit.fit.summary["p"]))
cphs.fits.boxplot_all_estimates(plot_name = "p-values", getter = lambda mdl_fit: mdl_fit.fit.summary["p"])
# %%
print(cphs.fits.table_all_estimates(table_name = "\n## Bias",       table_as_markdown=True, getter = lambda mdl_fit: mdl_fit.bias))
cphs.fits.boxplot_all_estimates(plot_name = "Bias", getter = lambda mdl_fit: mdl_fit.bias, marker_in_boxplot="berkson", dot_color="b")
# %%
print(cphs_small.fits.table_all_estimates(table_name = "\n## Estimates", table_as_markdown = True))
print(cphs_small.fits.table_all_estimates(table_name = "\n## Std. Error", table_as_markdown=True, getter = lambda mdl_fit: mdl_fit.fit.standard_errors_))
print(cphs_small.fits.table_all_estimates(table_name = "\n## p-values",   table_as_markdown=True, getter = lambda mdl_fit: mdl_fit.fit.summary["p"]))
print(cphs_small.fits.table_all_estimates(table_name = "\n## Bias",       table_as_markdown=True, getter = lambda mdl_fit: mdl_fit.bias))


# %%
# Linear model fit 
lms = LMList(voe_true, formula = "LBXT4 ~ RIDAGEYR + bmi + LBXTC")
lms.add_fit(voe_berkson.name, voe_berkson)
lms.add_fit(voe_epit.name, voe_epit)
lms.add_fit(voe_normal.name, voe_normal)
lms.add_fit(voe_lognormal.name, voe_lognormal)

# %%
print(lms.fits.table_all_estimates(table_name = "\n## Estimates", table_as_markdown = True))
# ! marker_in_boxplot uses name specified in table
lms.fits.boxplot_all_estimates(plot_name = "Estimates", marker_in_boxplot="reference")
# %%
print(lms.fits.table_all_estimates(table_name = "\n## Bias",       table_as_markdown=True, getter = lambda mdl_fit: mdl_fit.bias))
lms.fits.boxplot_all_estimates(plot_name = "Bias", getter = lambda mdl_fit: mdl_fit.bias, marker_in_boxplot="berkson", dot_color="b")
# lm_ref = fit_lm(voe_true.raw_data)
# lm_ref.params
# lm_berkson = fit_lm(voe_berkson.masked_data)
# lm_berkson.params
# lm_epit = fit_lm(voe_epit.masked_data)
# lm_normal = fit_lm(voe_normal.masked_data)
# lm_lognormal =  fit_lm(voe_lognormal.masked_data)

# %%
