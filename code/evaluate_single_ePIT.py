# This is a copy of "evaluate_single_BHMs", but with ePIT specific changes. 
# Since ePIT is not optiamlly implemented as of now, I would like to keep the ePIT specific differences 


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
from plotnine import ggplot, aes, geom_point, geom_abline

from datetime import date
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
data_path = r"../data/"

# %%
#!! ------------------------------- Parameters ---------------------------------------- !!#
### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###

# Specify the subset of variable extracted from the full data set
variable_subset = ["LBXT4", "RIDAGEYR", "bmi", "DR1TKCAL"]
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

voe_data = voe_data.loc[:, variable_subset].dropna(ignore_index = True)

# Data without error
voe = Data(
    name = "true", 
    raw_data = voe_data.dropna(ignore_index = True),
    error_type = "none"
)


data = voe.raw_data[variable_subset]

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
    response = response,
    error_cols = [],
    covariates = covariates,
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
#!! ------------------------ Define empirical KDE for BHMs ---------------------------- !!#
### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###

# Estimate the empirical density of the true covariates (i.e. an estimate for p(x)) 
# TODO: Improve this by ALOT; This is a lazy solution for now for proof of concept! 
from jax.scipy.stats import gaussian_kde
mdl = gaussian_kde(data.loc[:, covariates].values.T, bw_method = "scott")

# %% 
# #!! -------------------------- Create Error Data -------------------------------------- !!#
# ### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###
# Create the data with additive Gaussian error. 
# !!! To supply the normal_sd to JAX, it must be a FLOAT!
error_var = 0.3
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

voe_error= Data(
    name = f"epit_{error_var}", 
    raw_data = voe_data.dropna(ignore_index = True), 
    seed = 1234 + 2,
    error_vars = {"DR1TKCAL": jnp.array([error_var])}, 
    error_type="ePIT", 
    # Exclude the error on age and bmi for now to simplify the error structure
    cols_excluded_from_error = ["LBXT4", "RIDAGEYR", "bmi"],
    e_sigmoid = e_sigmoid, 
    e_inv_sigmoid = e_inv_sigmoid
)

plt.scatter(voe_error.raw_data.DR1TKCAL, voe_error.masked_data.DR1TKCAL)
plt.show()

# %%
ggplot_df = pd.concat([voe_error.raw_data[["DR1TKCAL"]], voe_error.masked_data[["DR1TKCAL"]]], axis = 1, keys = ["raw", "masked"])
ggplot_df.columns = ["raw", "masked"]
p = (
    ggplot(ggplot_df, aes(x = "raw", y = "masked")) + 
    geom_abline(slope = 1, intercept = 0, color = "red") + 
    geom_point() 
)
p.show()

print("Correlation:")
print(jnp.corrcoef(ggplot_df.raw.values, ggplot_df.masked.values))

# %%
# #!! -------------------------- Fit Naive Model --------------------------------------- !!#
# ### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###

# Naive Bayesian model NOT accounting for error; Parameters used as in model fit to compare to frequentistic OLS estimates
naive = BHM(
    data = voe_error.masked_data[["LBXT4", "RIDAGEYR", "bmi", "DR1TKCAL"]],
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
        "beta": jnp.repeat(
            jnp.array([frequentist_values]), 
            repeats = num_chains,
            axis = 0
        ),
        "log_sigma": jnp.repeat(jnp.log(jnp.var(jnp.asarray(data["LBXT4"].values))), num_chains, axis = 0)
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
# #!! -------------------------- Fit Corrected Model ------------------------------------ !!#
# ### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###
# To express the density for the measurement model as multivariate Gaussian, I need the covariance matrix, which is diagonal with the error variances on the diagonal. The error variances are the same across rows, but differ across covariates.
# Need to provide the (true) measurement variance. Decided to use a matrix to keep it flexible for different variances across covariates.
error_cov_matrix = jnp.diag(jnp.array([error_var])) # --> Working with only one error-affected covariate for now such that this is a scalar, but stick with the duck.

### The epit model is super cooked... 
# Epit Adjustments: 
# --- 1) Express data fully in terms of tilde(z) instead of tilde(x)
voe_error.masked_data.loc[:, "DR1TKCAL"] = jax.scipy.stats.norm.ppf(e_sigmoid(voe_error.masked_data.loc[:, "DR1TKCAL"].values)) 
# --- 2) Express KDE in terms of (true) z
# Because I draw z, the KDE must be expressed in terms of Z, too. 
# Basically, the design matrix is X, but I evaluate partially X and z where touched by error
data_raw_with_z = voe.raw_data
data_raw_with_z.loc[:, "DR1TKCAL"] = jax.scipy.stats.norm.ppf(e_sigmoid(voe_error.raw_data.loc[:, "DR1TKCAL"].values)) 
empirical_kde_mdl = gaussian_kde(data_raw_with_z.loc[:, covariates].values.T, bw_method = "scott")
# --- 3) Express observed values in terms of tilde(z) instead of tilde(x) for initial values
# ! When sub-selecting a columns, use this notation. Else, the code breaks.
init_vals = voe_error.masked_data[["DR1TKCAL"]].values
###

corrected = BHM(
    data = voe_error.masked_data, 
    response = "LBXT4",
    covariates = ["RIDAGEYR", "bmi", "DR1TKCAL"],
    # JAX does not allow me to pass error_cols as string to index the column in the design matrix touched by error. 
    # Because internally, the design matrix is a jnp array and not a dataframe, I need to pass the column index instead of the column name.
    error_cols = ["DR1TKCAL"], 
    post_log_dens = post_log_dens_epit,
    hyperparams = {
        "b": 100, # TODO: Depends on scale, of course. e.g. age reaches up to 80, calories reach up to 4k, bmi reaches up to 40; this is the SD of the posterior
        "c": 1,
        "d": 1
    },
    empirical_kde_mdl = empirical_kde_mdl,
    error_cov_matrix = error_cov_matrix,
    initial_positions = {
        # Beta estimates
        "beta": jnp.repeat(
            jnp.array([[0., 0., 0., 0.]]), 
            # jnp.array([frequentist_values]), 
            repeats = num_chains,
            axis = 0
        ),
        # Variance of Y, unknown --> Use empirical estimate
        "log_sigma": jnp.repeat(jnp.log(jnp.var(jnp.asarray(data["LBXT4"].values))), num_chains, axis = 0), 
        # For true observed values, start off with the ERROR-CONTAMINATED values only
        "Z_true": jnp.tile(init_vals, (num_chains, 1, 1)) 
    }, 
    inverse_mass_matrix = jnp.eye(3),
    rng_key = rng_key,
    num_chains = num_chains,
    inital_step_size = 1e-3, 
    warmup_steps = 100, 
    n_samples = 100, 
    e_sigmoid = e_sigmoid, 
    e_inv_sigmoid = e_inv_sigmoid
)
corrected.fit()
corrected.viz_chains(param_name = "beta")
corrected.viz_chains(param_name = "log_sigma")

# %%
abs_bias_corrected = corrected.mean_estimates(param_name = "beta") -  frequentist_values # jnp.array([8.7095, 0.00186197, -0.0748638, -0.000378365]))
print("Point Estimates:", corrected.mean_estimates(param_name = "beta") )
print("Bias: ", abs_bias_corrected) # Deviation (relative)
print("Relative Bias: ", (corrected.mean_estimates(param_name = "beta") - frequentist_values) / frequentist_values)

# %%
