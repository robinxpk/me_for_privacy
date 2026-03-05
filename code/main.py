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
from jax.scipy.stats import gaussian_kde
# To run code in parallel
import multiprocessing as mp
# !! Current implementation fits jax models in parallel
# !! Make sure they are based on spawn method NOT fork!
ctx = mp.get_context("spawn")

from datetime import date
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
data_path = r"../data/"

#!! ------------------------------- Functions ----------------------------------------- !!#
### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###
def single_iteration(error_name, error_variance, b): 
    # A single interation consists of: 
    # 1) Create a new data frame
    # 2) Fit a naive BHM
    # 3) Fit a BHM accounting for ME 
    # 4) Save results in csv file

    # The b-th data set is uses default seed + b. Rest is either given or constant. 
    voe_data = Data(
        name = f"{error_name}_{list(error_variance.values())[0]}", 
        raw_data = data, 
        seed = 1234 + b,
        error_vars = error_variance, 
        error_type = error_name,
        cols_excluded_from_error = ["LBXT4", "RIDAGEYR", "bmi"]
    )
    
    # Fit native Bayesian Hierarchical Model
    naive = BHM(
        data = voe_data.masked_data,
        response = response,
        error_cols = [],
        covariates = covariates,
        post_log_dens = post_log_dens,
        hyperparams = {
            "b": 100, # TODO: Depends on scale, of course. e.g. age reaches up to 80, calories reach up to 4k, bmi reaches up to 40; this is the SD of the posterior
            "c": 1,
            "d": 1
        },
        initial_positions = {
            "beta": jnp.zeros((num_chains, p)),
            "log_sigma": jnp.ones((num_chains, ))
        }, 
        empirical_kde_mdl = dummy_empirical_kde_mdl,
        error_cov_matrix = jnp.diag(jnp.array([])), 
        inverse_mass_matrix = jnp.eye(3),
        rng_key = rng_key,
        num_chains = num_chains,
        inital_step_size = 1e-3,
        # Burnin and warmup are default values with 1_000
        n_samples = n_samples
    )

    corrected = BHM(
        data = voe_data.masked_data,
        response = response,
        covariates = covariates,
        # JAX does not allow me to pass error_cols as string to index the column in the design matrix touched by error. 
        # Because internally, the design matrix is a jnp array and not a dataframe, I need to pass the column index instead of the column name.
        error_cols = error_subset, 
        post_log_dens = corrected_post_log_dens[error_name],
        hyperparams = {
            "b": 100, # TODO: Depends on scale, of course. e.g. age reaches up to 80, calories reach up to 4k, bmi reaches up to 40; this is the SD of the posterior
            "c": 1,
            "d": 1
        },
        empirical_kde_mdl = empirical_kde_mdl,
        error_cov_matrix = jnp.diag(jnp.array(list(error_variance.values()))) ,
        initial_positions = {
            # Beta estimates
            "beta": jnp.zeros((num_chains, p)),
            # Use empirical variance of response as initial value
            "log_sigma": jnp.repeat(jnp.log(jnp.var(jnp.asarray(data[response].values))), num_chains, axis = 0), 
            # For true observed values, start off with the ERROR-CONTAMINATED values only
            "X_true": jnp.tile(data[error_subset].values, (num_chains, 1, 1)) 
        }, 
        inverse_mass_matrix = jnp.eye(3),
        rng_key = rng_key,
        num_chains = num_chains,
        inital_step_size = 1e-3, 
        warmup_steps = n_warmup_steps, 
        n_samples = n_samples 
    )

    naive.fit()
    corrected.fit()

    ## Save results
    # Get R hat
    naive_rhat_beta = blackjax.diagnostics.potential_scale_reduction(naive.res.position["beta"])
    naive_rhat_log_sigma = blackjax.diagnostics.potential_scale_reduction(naive.res.position["log_sigma"])
    corrected_rhat_beta = blackjax.diagnostics.potential_scale_reduction(corrected.res.position["beta"])
    corrected_rhat_log_sigma = blackjax.diagnostics.potential_scale_reduction(corrected.res.position["log_sigma"])

    # Hardcoded results. yey.
    res = pd.DataFrame(
        data = {
            # Metadata
            "error":            [error_name                                 ],
            "error_variance":   [float(list(error_variance.values())[0])    ],
            "b":                [int(b)                                     ],

            # Naive Model Estimates
            "naive_beta0":      [float(naive.mean_estimates(param_name = "beta")[0])    ], 
            "naive_beta1":      [float(naive.mean_estimates(param_name = "beta")[1])    ],
            "naive_beta2":      [float(naive.mean_estimates(param_name = "beta")[2])    ],
            "naive_beta3":      [float(naive.mean_estimates(param_name = "beta")[3])    ], 
            "naive_log_sigma":  [float(naive.mean_estimates(param_name = "log_sigma"))  ], 

            "naive_rhat_beta0":     [float(naive_rhat_beta[0])  ],
            "naive_rhat_beta1":     [float(naive_rhat_beta[1])  ],
            "naive_rhat_beta2":     [float(naive_rhat_beta[2])  ],
            "naive_rhat_beta3":     [float(naive_rhat_beta[3])  ],
            "naive_rhat_log_sigma": [float(naive_rhat_log_sigma)],

            # Corrected Model Estimates
            "corrected_beta0":      [float(corrected.mean_estimates(param_name = "beta")[0])    ], 
            "corrected_beta1":      [float(corrected.mean_estimates(param_name = "beta")[1])    ],
            "corrected_beta2":      [float(corrected.mean_estimates(param_name = "beta")[2])    ],
            "corrected_beta3":      [float(corrected.mean_estimates(param_name = "beta")[3])    ], 
            "corrected_log_sigma":  [float(corrected.mean_estimates(param_name = "log_sigma"))  ], 

            "corrected_rhat_beta0":     [float(corrected_rhat_beta[0])  ],
            "corrected_rhat_beta1":     [float(corrected_rhat_beta[1])  ],
            "corrected_rhat_beta2":     [float(corrected_rhat_beta[2])  ],
            "corrected_rhat_beta3":     [float(corrected_rhat_beta[3])  ],
            "corrected_rhat_log_sigma": [float(corrected_rhat_log_sigma)] 
        }
    )
    filename = f"{error_name}_{list(error_variance.values())[0]}_{b}.csv"
    res.to_csv(filename, sep = ";", index = False)


def build_args(error_name:str, error_variance:dict, B:int): 
    # The starmap function allows to pass tuples of function inputs in the multiprocessing steps, e.g.: 
    # pool.starmap(g, [(1, 10), (2, 20), (3, 30)])
    # g(1, 10), g(2, 20), g(3, 30)
    # This function builds the required tuples 
    
    # Output shape: 
    # [..., (b_i, error_name, error_variance), ...] --> order in which the function single_iteration expects it

    return [(error_name, error_variance, b) for b in range(B)]


def fit_data_in_parallel(error_name, error_variance, B): 
    # TODO: Write this function such that it applies the loop in parallel and save results in an output file
    # Create a total of B data sets and fit naive as well as corrected model

    workers = min(B, max(1, os.cpu_count()) or 1)
    with ctx.Pool(processes=workers) as pool: 
        args = build_args(error_name = error_name, error_variance = error_variance, B = B)
        # Automatically writes output
        pool.starmap(single_iteration, args)
# %%
#!! ------------------------------- Parameters ---------------------------------------- !!#
### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###

# Lower sampling settings to speed up grid runs across error settings.
B = 5
n_warmup_steps = 1_000
n_burnin = 0 # implicitly in warmup; TODO: Remove burnin from BHM class anyway
n_samples = 1_000

# Specify the subset of variable extracted from the full data set
variable_subset = ["LBXT4", "RIDAGEYR", "bmi", "DR1TKCAL"]
error_subset = ["DR1TKCAL"]
# Error variances which are iterated over
errors = ["normal", "lognormal", "ePIT"]
errors = ["normal", "lognormal"]
error_variances_by_error = {
    # The BHM expects a dictonary with the name of the error variance and the value. Thus, use a list of dictionaries for each error for differen error variance values
    "normal": [
        {"DR1TKCAL": 1},
        {"DR1TKCAL": 2}
    ],
    # ! Supply the variance of the NORMAL distribution, i.e. variance of log(error) ~ N(mu, var) --> See README for further details
    "lognormal": [
        {"DR1TKCAL": 1},
        {"DR1TKCAL": 2}
    ],
    "epit": [
        {"DR1TKCAL": 1},
        {"DR1TKCAL": 2}
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
num_chains = 3

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
dummy_empirical_kde_mdl = KDE_Dummy_Model()
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
            fit_data_in_parallel(error_name = error, error_variance = error_variance, B = B)
# %%
