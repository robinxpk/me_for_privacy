# %%
import os
# BEFORE important jax, set XLA flags to disable threading such that data sets can run in parallel (hopefully) without issues
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" 
# %%
import jax
import jax.numpy as jnp
import jax.nn as jnn
import blackjax
import matplotlib.pyplot as plt
import pandas as pd

from ME.BHM import BHM
from ME.KDE import KDE_Dummy_Model
from ME.Data import Data
from ME.functions import post_log_dens, post_log_dens_gaussian_additive, post_log_dens_lognormal_multiplicative, post_log_dens_epit
from ME.utils import fit_data_in_parallel
from jax.scipy.stats import gaussian_kde
# To run code in parallel
import multiprocessing as mp
# !! Current implementation fits jax models in parallel
# !! Make sure they are based on spawn method NOT fork!
ctx = mp.get_context("spawn")

from datetime import date
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
data_path = r"../data/"

#!! ------------------------------- Parameters ---------------------------------------- !!#
### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###
# TODO: Can I move this into a config (hydra)?
# Lower sampling settings to speed up grid runs across error settings.
B = 1
n_warmup_steps = 1
n_samples = 1

# Specify the subset of variable extracted from the full data set
variable_subset = ["LBXT4", "RIDAGEYR", "bmi", "DR1TKCAL"]
error_subset = ["DR1TKCAL"]
# Error variances which are iterated over
errors = ["ePIT", "lognormal", "normal"]
errors = ["normal", "lognormal"]

# ref_var_normal_error = voe.raw_data[error_subset].var()
ref_var_normal_error = 450411.083711
error_variances_by_error = {
    # The BHM expects a dictonary with the name of the error variance and the value. Thus, use a list of dictionaries for each error for differen error variance values
    "normal": [
        {"DR1TKCAL": 0.8 * ref_var_normal_error},
        {"DR1TKCAL": 3.5 * ref_var_normal_error}
    ],
    # ! Supply the variance of the NORMAL distribution, i.e. variance of log(error) ~ N(mu, var) --> See README for further details
    "lognormal": [
        {"DR1TKCAL": 0.1},
        {"DR1TKCAL": 0.3}
    ],
    "ePIT": [
        {"DR1TKCAL": 0.35},
        {"DR1TKCAL": 0.7}
    ]
}
# Specify the density functions to use
corrected_post_log_dens = {
    "normal": post_log_dens_gaussian_additive,
    "lognormal": post_log_dens_lognormal_multiplicative,
    "ePIT": post_log_dens_epit
}

response = "LBXT4"
covariates = ["RIDAGEYR", "bmi", "DR1TKCAL"]


# -1 for response variable, +1 for intercept; This is only kept for clarity
p = len(variable_subset) - 1 + 1
num_chains = 2

# %%
#!! ------------------------------- Load the Data ------------------------------------- !!#
### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###
voe_data = pd.read_csv(f"{data_path}voe_data.csv", sep = ";", header = 0)
factor_vars = (
    # -- Survival Indicator
    "MORTSTAT",
    # -- Exam sample weight (combined)
    "WTMEC4YR"
)
for col in factor_vars:
    voe_data[col] = voe_data[col].astype("category")
data = voe_data.drop("WTMEC4YR", axis = 1).dropna(ignore_index=True)[variable_subset]

# %% 
#!! ------------------------------- Specify empirical KDEs ---------------------------- !!#
### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###
# TODO: This is a lazy solution for an empirical KDE for now. Improve this! 
empirical_kde_mdl = gaussian_kde(data[covariates].values.T, bw_method = "scott")

# %% 
#!! ------------------------ Fit Error Models on multiple Error Data ------------------ !!#
### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###
if __name__ == "__main__": 
    # For each error (3 different errors)
    for error in errors: 
        # For each error variance within the corresponding error (2 variances each error for now)
        for error_variance in error_variances_by_error[error]:
            fit_data_in_parallel(
                error_name = error, 
                error_variance = error_variance, 
                B = B, 
                empirical_kde_mdl = empirical_kde_mdl, 

                raw_data = data, 
                num_chains = num_chains, 
                covariates = covariates,
                response = response,
                error_subset = error_subset,
                p = p,
                n_samples = n_samples,
                n_warmup_steps = n_warmup_steps, 
                corrected_post_log_dens = corrected_post_log_dens, 

                rng_key = rng_key, 
                ctx = ctx
            )
