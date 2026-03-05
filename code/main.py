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

# %%
#!! ------------------------------- Parameters ---------------------------------------- !!#
### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###

# Lower sampling settings to speed up grid runs across error settings.
frequentist_values =  jnp.array([8.7095, 0.00186197, -0.0748638, -0.000378365])
B = 1
n_warmup_steps = 1000
n_burnin = 0 # implicitly in warmup
n_samples = 100

# Specify the subset of variable extracted from the full data set
variable_subset = ["LBXT4", "RIDAGEYR", "bmi", "DR1TKCAL"]
error_subset = ["DR1TKCAL"]
# Error variances which are iterated over
errors = ["normal", "lognormal", "ePIT"]
error_variances_by_error = {
    # The BHM expects a dictonary with the name of the error variance and the value. Thus, use a list of dictionaries for each error for differen error variance values
    "normal": [
        {"DR1TKCAL": 1},
        {"DR1TKCAL": 2}
    ],
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
#!! ----------------------------------------------------------------------------------- !!#
### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###
def create_and_fit_on_one_dataset(b, error_name, error_variance): 
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

    ## Get results
    # Get Bias
    abs_bias_naive = naive.mean_estimates(param_name = "beta") -  frequentist_values # jnp.array([8.7095, 0.00186197, -0.0748638, -0.000378365]))
    rel_bias_naive = (naive.mean_estimates(param_name = "beta") - frequentist_values) / frequentist_values
    abs_bias_corrected = corrected.mean_estimates(param_name = "beta") -  frequentist_values # jnp.array([8.7095, 0.00186197, -0.0748638, -0.000378365]))
    rel_bias_corrected = (corrected.mean_estimates(param_name = "beta") - frequentist_values) / frequentist_values
    # Get R hat
    rhat_beta_naive = blackjax.diagnostics.potential_scale_reduction(naive.res.position["beta"])
    rhat_log_sigma_naive = blackjax.diagnostics.potential_scale_reduction(naive.res.position["log_sigma"])
    rhat_beta_corrected = blackjax.diagnostics.potential_scale_reduction(corrected.res.position["beta"])
    rhat_log_sigma_corrected = blackjax.diagnostics.potential_scale_reduction(corrected.res.position["log_sigma"])

    # TODO: Save results somewhere in output files

    return {
        "error": error_name,
        "error_variance": float(list(error_variance.values())[0]),
        "b": int(b),
        "abs_bias_naive": [float(x) for x in abs_bias_naive],
        "rel_bias_naive": [float(x) for x in rel_bias_naive],
        "abs_bias_corrected": [float(x) for x in abs_bias_corrected],
        "rel_bias_corrected": [float(x) for x in rel_bias_corrected],
        "rhat_beta_naive": [float(x) for x in rhat_beta_naive],
        "rhat_log_sigma_naive": float(rhat_log_sigma_naive),
        "rhat_beta_corrected": [float(x) for x in rhat_beta_corrected],
        "rhat_log_sigma_corrected": float(rhat_log_sigma_corrected),
    }


def fit_data_in_parallel(error_name, error_variance, B): 
    # TODO: Write this function such that it applies the loop in parallel and save results in an output file
    # Create a total of B data sets and fit naive as well as corrected model

    workers = min(B, max(1, os.cpu_count()) or 1)
    with ctx.Pool(processes=workers) as pool: 
        rows = pool.map(create_and_fit_on_one_dataset, TODO)

    out_dir = os.path.join(data_path, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"fit_results_{error_name}_{list(error_variance.values())[0]}.csv")
    pd.DataFrame(rows).to_csv(out_file, index=False)



# TODO: The following is nasty. So nasty. To what degree can I parallelize? The chains will be run parallel using jax, but can I additionally parallleize the data sets?
# TODO: Fix those for loop into parallel. Cooked.
# For each error (3 different errors)
for error in errors: 
    # For each error variance within the corresponding error (2 variances each error for now)
    for error_variance in error_variances_by_error[error]:
        fit_data_in_parallel(error_name = error, error_variance = error_variance, B = B)



