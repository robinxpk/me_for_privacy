# %%
import jax
import jax.numpy as jnp
import jax.nn as jnn
import blackjax
import matplotlib.pyplot as plt
import pandas as pd

from ME.BHM import BHM
from ME.Data import Data

from datetime import date
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
data_path = r"../data/"

# This is the native density bzw. true density of the model if no error is applied
def post_log_dens(y, X, params, b, c, d):
    ### --- params is a dictionary containing the current value of the parameters
    ### --- Assign dictionary entries to variables 
    beta = params["beta"]
    log_sigma = params["log_sigma"]
    sigma = jnp.exp(log_sigma)
    ### --- Identify dimensions of coefficient vector
    n = X.shape[0]
    eta = X @ beta

    # ! Density is expressed in terms of log_sigma seen from last summand in log_sigma_prior_term
    log_likelihood_term = - n / 2 * jnp.log(sigma ** 2) - 1 / (2 * sigma ** 2) * jnp.sum((y - eta) ** 2)
    log_beta_prior_term = - 1 / (2 * b ** 2) * jnp.sum(beta ** 2)
    log_sigma_prior_term = (c - 1) * jnp.log(sigma ** 2) - d * sigma ** 2 + jnp.log(sigma ** 2) 
    return log_likelihood_term.sum() + log_beta_prior_term + log_sigma_prior_term 

# %%
# load Data
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
# Create error attached Data objects 
normal_sd = 10 ** 3 # NOTE: Depends on scale, of course. e.g. calories reach up to 4k
voe_true = Data(
    name = "true", 
    raw_data = voe_data.dropna(ignore_index = True),
    prob = 0, 
    error_type = "berkson", 
    cluster_based = False
)


# %%
# Compare base model with frequentist OLS estimates
data = voe_true.raw_data[["LBXT4", "RIDAGEYR", "bmi", "DR1TKCAL"]]
# -1 for response variable, +1 for intercept; This is only kept for clarity
p = data.shape[1] - 1 + 1
num_chains = 4

bhm = BHM(
    data = data, 
    response = "LBXT4",
    covariates = ["RIDAGEYR", "bmi", "DR1TKCAL"],
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
    inverse_mass_matrix = jnp.eye(3),
    rng_key = rng_key,
    num_chains = num_chains,
    inital_step_size = 1e-3
)
# %%
# Creates bhm.res 
bhm.fit()
# %%
bhm.viz_chains(param_name = "beta")
bhm.mean_estimates(param_name = "beta")
# %%
bhm.viz_chains(param_name = "log_sigma")
bhm.mean_estimates(param_name = "log_sigma")
# %%
# Frequentist OLS estimates for comparison
## Estimates
# |           |    Frequent. |
# |:----------|-------------:|
# | Intercept |  8.7095      |
# | RIDAGEYR  |  0.00186197  |
# | bmi       | -0.0748638   |
# | DR1TKCAL  | -0.000378365 |
# %% 
# Deviation (absolute)
bhm.mean_estimates(param_name = "beta") - jnp.array([8.7095, 0.00186197, -0.0748638, -0.000378365])
# %% 
# Deviation (relative)
(bhm.mean_estimates(param_name = "beta") - jnp.array([8.7095, 0.00186197, -0.0748638, -0.000378365])) / jnp.array([8.7095, 0.00186197, -0.0748638, -0.000378365]) 
# %%
###
# Quantify bias due to measurement error
###
normal_sd = 10 ** 3 # NOTE: Depends on scale, of course. e.g. calories reach up to 4k
voe_normal= Data(
    name = f"normal_{normal_sd}", 
    raw_data = voe_data.dropna(ignore_index = True), 
    prob = 1, 
    error_factors = jnp.array([normal_sd]), 
    error_type="normal"
)
lognormal_sd = 1
voe_lognormal = Data(
    name = f"lognormal_{lognormal_sd}", 
    raw_data = voe_data.dropna(ignore_index = True), 
    prob = 1, 
    error_factors = jnp.array([lognormal_sd]), 
    error_type="lognormal"
)

epit_sd = 1
voe_epit = Data(
    name = f"epit_{epit_sd}", 
    raw_data = voe_data.dropna(ignore_index = True), 
    prob = 1, 
    error_factors = jnp.array([epit_sd]), 
    error_type="ePIT"
)

# %%
# Quantify model bias
def get_data_for_model(
        seed, 
        name, raw_data, prob, error_factors, error_type, cols_excluded_from_error = None, 
        cols = ["LBXT4", "RIDAGEYR", "DR1TKCAL", "bmi"]
): 
    voe = Data(
        name = name, 
        raw_data = raw_data, 
        prob = prob, 
        error_factors = error_factors, 
        error_type=error_type,
        cols_excluded_from_error=cols_excluded_from_error
    )
    return voe.masked_data[cols]

def log_corrected_density(): 
    pass

def quantify_bias(
        raw_data, 
        error_type, error_factor, cols_excluded_from_error,
        log_corrected_density, rng_key, 
        B = 100
):
    bias_estimates = {"naive": [], "corrected": []}
    for b in range(B):
        rng_key, _ = jax.random.split(rng_key)
        data = get_data_for_model(
            seed=b, 
            name=f"bias_{b}", 
            raw_data=raw_data, 
            prob=1, 
            error_factors=jnp.array([error_factor]), 
            error_type=error_type, 
            cols_excluded_from_error=cols_excluded_from_error
        )
        # Naive Bayesian model not accounting for error; Parameters used as in model fit to compare to frequentistic OLS estimates
        naive = BHM(
            data = data, 
            response = "LBXT4",
            covariates = ["RIDAGEYR", "bmi", "DR1TKCAL"],
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
            inverse_mass_matrix = jnp.eye(3),
            rng_key = rng_key,
            num_chains = num_chains,
            inital_step_size = 1e-3
        )
        naive.fit()
        # Bias is then the deviation of the naive model's estimates from the true parameters (or frequentistic OLS estimates)
        naive_bias_estimate = naive.mean_estimates(param_name = "beta") - bhm.mean_estimates(param_name = "beta")
        bias_estimates["naive"].append(naive_bias_estimate)


        # corrected = BHM(
        #     data = data, 
        #     response = "LBXT4",
        #     covariates = ["RIDAGEYR", "bmi", "DR1TKCAL"],
        #     post_log_dens = log_corrected_density,
        #     hyperparams = {
        #         "b": 100, # TODO: Depends on scale, of course. e.g. age reaches up to 80, calories reach up to 4k, bmi reaches up to 40; this is the SD of the posterior
        #         "c": 1,
        #         "d": 1
        #     },
        #     initial_positions = {
        #         "beta": jnp.zeros((num_chains, p)),
        #         "log_sigma": jnp.ones((num_chains, ))
        #     }, 
        #     inverse_mass_matrix = jnp.eye(3),
        #     rng_key = rng_key,
        #     num_chains = num_chains,
        #     inital_step_size = 1e-3
        # )
        # corrected.fit()
        # # Bias is then the deviation of the corrected model's estimates from the true parameters (or frequentistic OLS estimates)
        # corrected_bias_estimate = corrected.mean_estimates(param_name = "beta") - bhm.mean_estimates(param_name = "beta")
        # bias_estimates["corrected"].append(corrected_bias_estimate)
    return bias_estimates

quantify_bias(
    raw_data = voe_data.dropna(ignore_index = True), 
    error_type = "normal", error_factor = normal_sd, cols_excluded_from_error = None,
    log_corrected_density = log_corrected_density,
    B = 2, rng_key = rng_key
)

# %% 
# For Berkson error, we can only use the data once, since the error is cluster-based and thus not i.i.d. across rows.
voe_berkson = Data(
    name = "berkson", 
    raw_data = voe_data.dropna(ignore_index = True),
    prob = 1, 
    error_type = "berkson", 
    cluster_based = True, 
    cols_excluded_from_error = ["PERMTH_EXM"]
)