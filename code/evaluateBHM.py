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
from ME.functions import post_log_dens, post_log_dens_gaussian_additive

from datetime import date
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
data_path = r"../data/"


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
voe_data = voe_data.drop("WTMEC4YR", axis = 1)

# Data without error
voe_true = Data(
    name = "true", 
    raw_data = voe_data.dropna(ignore_index = True),
    prob = 0, 
    error_type = "berkson", 
    cluster_based = False
)

data = voe_true.raw_data[["LBXT4", "RIDAGEYR", "bmi", "DR1TKCAL"]]
# -1 for response variable, +1 for intercept; This is only kept for clarity
p = data.shape[1] - 1 + 1
num_chains = 4

# %%
#!! -------------------------- Compare Bayes vs Freq Fit------------------------------- !!#
### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###
# First, compare the Bayesian model fit to the frequentistic OLS estimates to check if the model is correctly specified and implemented.

# Define a dummy KDE model. More on it later.
# The dummy one is basically how the KDE is ignored when evaluating the density of the error-free data. 
dummy_empirical_kde_mdl = KDE_Dummy_Model()

# Define the BHM and fit it.
bhm = BHM(
    data = data, 
    response = "LBXT4",
    error_cols = [],
    covariates = ["RIDAGEYR", "bmi", "DR1TKCAL"],
    post_log_dens = post_log_dens,
    hyperparams = {
        "b": 100, # TODO: Depends on scale, of course. e.g. age reaches up to 80, calories reach up to 4k, bmi reaches up to 40; this is the SD of the posterior
        "c": 1,
        "d": 1
    },
    error_cov_matrix = jnp.diag(jnp.array([])),
    empirical_kde_mdl = dummy_empirical_kde_mdl,
    initial_positions = {
        "beta": jnp.zeros((num_chains, p)),
        "log_sigma": jnp.ones((num_chains, ))
    }, 
    inverse_mass_matrix = jnp.eye(3),
    rng_key = rng_key,
    num_chains = num_chains,
    inital_step_size = 1e-3
)

bhm.fit()
# %%
# Some visualization and comparison to frequentistic OLS estimates to check if the model is correctly specified and implemented.
# Beta estimates
bhm.viz_chains(param_name = "beta")
print(bhm.mean_estimates(param_name = "beta"))
# Sigma estimate
bhm.viz_chains(param_name = "log_sigma")
print(bhm.mean_estimates(param_name = "log_sigma"))
# %%
# Frequentist OLS estimates for comparison
## Estimates
# |           |    Frequent. |
# |:----------|-------------:|
# | Intercept |  8.7095      |
# | RIDAGEYR  |  0.00186197  |
# | bmi       | -0.0748638   |
# | DR1TKCAL  | -0.000378365 |
# Deviation (absolute)
frequentist_values =  jnp.array([8.7095, 0.00186197, -0.0748638, -0.000378365])
abs_bias = bhm.mean_estimates(param_name = "beta") -  frequentist_values # jnp.array([8.7095, 0.00186197, -0.0748638, -0.000378365]))
print("Bias: ", abs_bias) # Deviation (relative)

print("Relative Bias: ", (bhm.mean_estimates(param_name = "beta") - frequentist_values) / frequentist_values)
# %%
#!! -------------------------- Quantify Bias due to ME -------------------------------- !!#
### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###

# Estimate the empirical density of the true covariates (i.e. an estimate for p(x)) 
# TODO: Improve this by ALOT; This is a lazy solution for now for proof of concept! 
from jax.scipy.stats import gaussian_kde
mdl = gaussian_kde(voe_true.raw_data[["RIDAGEYR", "bmi", "DR1TKCAL"]].values.T, bw_method = "scott")

# %% 
# --->> #!! -------------------------- GAUSSIAN ADDITIVE ME ----------------------------------- !!#
# --->> ### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###

# TODO: Define some loop-function which I call here where
# 1) Create new data 
# 2) Fit the naive model and get bias estimate
# 3) Fit the corrected model and get bias estimate
# 4) Evaluate 

# Create the data with additive Gaussian error. 
# !!! To supply the normal_sd to JAX, it must be a FLOAT!
normal_sd = 10. ** 3# NOTE: Depends on scale, of course. e.g. calories reach up to 4k
voe_normal= Data(
    name = f"normal_{normal_sd}", 
    raw_data = voe_data.dropna(ignore_index = True), 
    prob = 1, 
    error_factors = jnp.array([normal_sd]), 
    error_type="normal", 
    # Exclude the error on age and bmi for now to simplify the error structure
    cols_excluded_from_error = ["RIDAGEYR", "bmi"]
)

# %%
# Single model fits
## The naive model 
# Naive Bayesian model not accounting for error; Parameters used as in model fit to compare to frequentistic OLS estimates
naive = BHM(
    data = voe_normal.masked_data[["LBXT4", "RIDAGEYR", "bmi", "DR1TKCAL"]],
    response = "LBXT4",
    error_cols = [],
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
    empirical_kde_mdl = dummy_empirical_kde_mdl,
    error_cov_matrix = jnp.diag(jnp.array([])), 
    inverse_mass_matrix = jnp.eye(3),
    rng_key = rng_key,
    num_chains = num_chains,
    inital_step_size = 1e-3
)
naive.fit()
naive.viz_chains(param_name = "beta")

abs_bias_naive = naive.mean_estimates(param_name = "beta") -  frequentist_values # jnp.array([8.7095, 0.00186197, -0.0748638, -0.000378365]))
print("Bias: ", abs_bias_naive) # Deviation (relative)

print("Relative Bias: ", (naive.mean_estimates(param_name = "beta") - frequentist_values) / frequentist_values)
# %%
# To express the density for the measurement model as multivariate Gaussian, I need the covariance matrix, which is diagonal with the error variances on the diagonal. The error variances are the same across rows, but differ across covariates.
# Need to provide the (true) measurement variance. Decided to use a matrix to keep it flexible for different variances across covariates.
error_cov_matrix = jnp.diag(jnp.array([normal_sd])) # --> Working with only one error-affected covariate for now such that this is a scalar, but stick with the duck.

# TODO: Need to apply different variances to the variables. Make the Code work with a dictionary or something which contains variable specific variances.
# TODO: My normal_sd is actually the variance. Rework how the error is added anyway... 
# TODO: Just realized that sigma in the sampler is the variance. OMG. Fix those namings. Horror....
# Old sigmas; TODOs remain! - Do not want to deal with naming issues. 
# "log_sigma_age": jnp.repeat(jnp.log(jnp.array([normal_sd])), num_chains, axis = 0),
# "log_sigma_bmi": jnp.repeat(jnp.log(jnp.array([normal_sd])), num_chains, axis = 0),
# "log_sigma_kcal": jnp.repeat(jnp.log(jnp.array([normal_sd])), num_chains, axis = 0),

corrected = BHM(
    data = data, 
    response = "LBXT4",
    covariates = ["RIDAGEYR", "bmi", "DR1TKCAL"],
    # JAX does not allow me to pass error_cols as string to index the column in the design matrix touched by error. 
    # Because internally, the design matrix is a jnp array and not a dataframe, I need to pass the column index instead of the column name.
    error_cols = ["DR1TKCAL"], 
    post_log_dens = post_log_dens_gaussian_additive,
    hyperparams = {
        "b": 100, # TODO: Depends on scale, of course. e.g. age reaches up to 80, calories reach up to 4k, bmi reaches up to 40; this is the SD of the posterior
        "c": 1,
        "d": 1
    },
    empirical_kde_mdl = mdl,
    error_cov_matrix = error_cov_matrix,
    initial_positions = {
        ## Ignorant starting values
        # "beta": jnp.zeros((num_chains, p)),
        # "log_sigma": jnp.ones((num_chains, )), 
        # "log_sigma_age": jnp.ones((num_chains, )),
        # "log_sigma_bmi": jnp.ones((num_chains, )),
        # "log_sigma_kcal": jnp.ones((num_chains, )),
        ## Frequentist point estimates 
        # Beta estimates
        "beta": jnp.repeat(
            jnp.array([frequentist_values]), 
            repeats = num_chains,
            axis = 0
        ),
        # Variance of Y, unknown... 
        "log_sigma": jnp.repeat(jnp.log(jnp.var(jnp.asarray(data["LBXT4"].values))), num_chains, axis = 0), 
        # For true observed values, start off with the ERROR-CONTAMINATED values only
        "X_true": jnp.tile(data[["DR1TKCAL"]].values, (num_chains, 1, 1)) 
    }, 
    inverse_mass_matrix = jnp.eye(3),
    rng_key = rng_key,
    num_chains = num_chains,
    inital_step_size = 1e-3, 
    burnin = 1000,
    warmup_steps = 1000, 
    n_samples = 1000 
)
corrected.fit()
corrected.viz_chains(param_name = "beta")

# # %%
# # Run the loop:
# # Quantify model bias
# def get_data_for_model(
#         seed, 
#         name, raw_data, prob, error_factors, error_type, cols_excluded_from_error = None, 
#         cols = ["LBXT4", "RIDAGEYR", "DR1TKCAL", "bmi"]
# ): 
#     voe = Data(
#         name = name, 
#         raw_data = raw_data, 
#         prob = prob, 
#         error_factors = error_factors, 
#         error_type=error_type,
#         cols_excluded_from_error=cols_excluded_from_error,
#         seed = seed
#     )
#     return voe.masked_data[cols]

# def quantify_bias(
#         raw_data, 
#         error_type, error_factor, cols_excluded_from_error,
#         log_corrected_density, rng_key, 
#         empirical_kde_mdl,
#         B = 100, 
#         n_burnin = 100, n_warmup_steps = 100, n_samples = 100
# ):
#     bias_estimates = {"naive": [], "corrected": []}
#     for b in range(B):
#         rng_key, _ = jax.random.split(rng_key)
#         data = get_data_for_model(
#             seed=b, 
#             name=f"bias_{b}", 
#             raw_data=raw_data, 
#             prob=1, 
#             error_factors=jnp.array([error_factor]), 
#             error_type=error_type, 
#             cols_excluded_from_error=cols_excluded_from_error
#         )
#         # # Naive Bayesian model not accounting for error; Parameters used as in model fit to compare to frequentistic OLS estimates
#         # naive = BHM(
#         #     data = data, 
#         #     response = "LBXT4",
#         #     covariates = ["RIDAGEYR", "bmi", "DR1TKCAL"],
#         #     post_log_dens = post_log_dens,
#         #     hyperparams = {
#         #         "b": 100, # TODO: Depends on scale, of course. e.g. age reaches up to 80, calories reach up to 4k, bmi reaches up to 40; this is the SD of the posterior
#         #         "c": 1,
#         #         "d": 1
#         #     },
#         #     initial_positions = {
#         #         "beta": jnp.zeros((num_chains, p)),
#         #         "log_sigma": jnp.ones((num_chains, ))
#         #     }, 
#         #     empirical_kde_mdl = dummy_empirical_kde_mdl,
#         #     inverse_mass_matrix = jnp.eye(3),
#         #     rng_key = rng_key,
#         #     num_chains = num_chains,
#         #     inital_step_size = 1e-3
#         # )
#         # naive.fit()
#         # # Bias is then the deviation of the naive model's estimates from the true parameters (or frequentistic OLS estimates)
#         # naive_bias_estimate = naive.mean_estimates(param_name = "beta") - bhm.mean_estimates(param_name = "beta")
#         # bias_estimates["naive"].append(naive_bias_estimate)


#         corrected = BHM(
#             data = data, 
#             response = "LBXT4",
#             covariates = ["RIDAGEYR", "bmi", "DR1TKCAL"],
#             post_log_dens = log_corrected_density,
#             hyperparams = {
#                 "b": 100, # TODO: Depends on scale, of course. e.g. age reaches up to 80, calories reach up to 4k, bmi reaches up to 40; this is the SD of the posterior
#                 "c": 1,
#                 "d": 1,
#                 "e": 1,
#                 "f": 1
#             },
#             empirical_kde_mdl = empirical_kde_mdl,
#             initial_positions = {
#                 ## Ignorant starting values
#                 # "beta": jnp.zeros((num_chains, p)),
#                 # "log_sigma": jnp.ones((num_chains, )), 
#                 # "log_sigma_age": jnp.ones((num_chains, )),
#                 # "log_sigma_bmi": jnp.ones((num_chains, )),
#                 # "log_sigma_kcal": jnp.ones((num_chains, )),
#                 ## Frequentist point estimates 
#                 # Beta estimates
#                 "beta": jnp.repeat(
#                     jnp.array([frequentist_values]), 
#                     repeats = num_chains,
#                     axis = 0
#                 ),
#                 # Variance of Y, unknown... 
#                 # TODO: Use empirical estimate of true values. lol. wtf.
#                 "log_sigma": jnp.ones((num_chains, )), 
#                 # Error variances known from above
#                 # TODO: Need to apply different variances to the variables. Make the Code work with a dictionary or something which contains variable specific variances.
#                 # TODO: My normal_sd is actually the variance. Rework how the error is added anyway... 
#                 # TODO: Just realized that sigma in the sampler is the variance. OMG. Fix those namings. Horror....
#                 "log_sigma_age": jnp.repeat(jnp.log(jnp.array([normal_sd])), num_chains, axis = 0),
#                 "log_sigma_bmi": jnp.repeat(jnp.log(jnp.array([normal_sd])), num_chains, axis = 0),
#                 "log_sigma_kcal": jnp.repeat(jnp.log(jnp.array([normal_sd])), num_chains, axis = 0),
#                 # For true observed values, start off with the ERROR-CONTAMINATED values
#                 "X_true": jnp.tile(data[["RIDAGEYR", "bmi", "DR1TKCAL"]].values, (num_chains, 1, 1)) 
#             }, 
#             inverse_mass_matrix = jnp.eye(3),
#             rng_key = rng_key,
#             num_chains = num_chains,
#             inital_step_size = 1e-3, 
#             burnin = n_burnin,
#             warmup_steps = n_warmup_steps, 
#             n_samples = n_samples 
#         )
#         corrected.fit()
#         # Bias is then the deviation of the corrected model's estimates from the true parameters (or frequentistic OLS estimates)
#         corrected_bias_estimate = corrected.mean_estimates(param_name = "beta") - bhm.mean_estimates(param_name = "beta")
#         bias_estimates["corrected"].append(corrected_bias_estimate)
#     return corrected
#     return bias_estimates

# test = quantify_bias(
#     raw_data = voe_data.dropna(ignore_index = True), 
#     error_type = "normal", error_factor = normal_sd, cols_excluded_from_error = None,
#     log_corrected_density = post_log_dens_gaussian_additive,
#     B = 1, rng_key = rng_key, empirical_kde_mdl = mdl, 
#     n_burnin = 1000, n_warmup_steps = 1000, n_samples = 10000
# )


# # %%
# # --->> ### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###
# # --->> #!! ------------------------------- LOGNORMAL ME -------------------------------------- !!#
# # --->> ### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###
# lognormal_sd = 1
# voe_lognormal = Data(
#     name = f"lognormal_{lognormal_sd}", 
#     raw_data = voe_data.dropna(ignore_index = True), 
#     prob = 1, 
#     error_factors = jnp.array([lognormal_sd]), 
#     error_type="lognormal"
# )


# # --->> ### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###
# # --->> #!! ------------------------------- ePIT ME -------------------------------------- !!#
# # --->> ### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###
# epit_sd = 1
# voe_epit = Data(
#     name = f"epit_{epit_sd}", 
#     raw_data = voe_data.dropna(ignore_index = True), 
#     prob = 1, 
#     error_factors = jnp.array([epit_sd]), 
#     error_type="ePIT"
# )

# # --->> ### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###
# # --->> #!! ------------------------------- Berkson ME ----------------------------------- !!#
# # --->> ### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###
# # For Berkson error, we can only use the data once, since the error is cluster-based and thus not i.i.d. across rows.
# voe_berkson = Data(
#     name = "berkson", 
#     raw_data = voe_data.dropna(ignore_index = True),
#     prob = 1, 
#     error_type = "berkson", 
#     cluster_based = True, 
#     cols_excluded_from_error = ["PERMTH_EXM"]
# )

# %%
