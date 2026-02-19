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
def post_log_dens(y, X, params, empirical_logdensity, b, c, d):
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
    log_empirical_density = empirical_logdensity(X).sum()
    return log_likelihood_term.sum() + log_empirical_density + log_beta_prior_term + log_sigma_prior_term 

def dummy_empirical_logdensity(X): 
    # This "empirical (log-)density" functions returns constants 0s, such that it drops from the log density sum. 
    # This is allows to easily switch between using an empirical density or not, without having to change the structure of the log density function.
    return jnp.zeros(X.shape[0])

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
    empirical_logdensity = dummy_empirical_logdensity,
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
# %% 
# Estimate the empirical density of the true covariates (i.e. an estimate for p(x)) 
# TODO: Improve this by ALOT; This is a lazy solution for now for proof of concept! 
from sklearn.mixture import GaussianMixture
mdl = GaussianMixture(n_components = 5, covariance_type = "full", random_state = 0).fit(X = voe_true.raw_data[["RIDAGEYR", "bmi", "DR1TKCAL"]].values)
def gmm_empirical_logdensity(X, mdl): 
    return jnp.array(mdl.score_samples(X))
lazy_empirical_logdensity = lambda X: gmm_empirical_logdensity(X, mdl)

# %% 
# Gaussian, addtive error
normal_sd = 10 ** 3 # NOTE: Depends on scale, of course. e.g. calories reach up to 4k
voe_normal= Data(
    name = f"normal_{normal_sd}", 
    raw_data = voe_data.dropna(ignore_index = True), 
    prob = 1, 
    error_factors = jnp.array([normal_sd]), 
    error_type="normal"
)

def post_log_dens_gaussian_additive(y, X, params, empirical_logdensity, b, c, d, e, f):
    ### --- params is a dictionary containing the current value of the parameters
    ### --- Assign dictionary entries to variables 
    beta = params["beta"]
    log_sigma = params["log_sigma"]
    log_sigma_age = params["log_sigma_age"]
    log_sigma_bmi = params["log_sigma_bmi"]
    log_sigma_kcal = params["log_sigma_kcal"]
    # To extract the true observed values, I need the design matrix only containing actual covariates, i.e. without intercept
    X_no_intercept = X[:, 1:] 
    X_true = params["X_true"]
    # To express the density for the measurement model as multivariate Gaussian, I need the covariance matrix, which is diagonal with the error variances on the diagonal. The error variances are the same across rows, but differ across covariates.
    error_cov_matrix = jnp.diag(jnp.exp(jnp.array([log_sigma_age, log_sigma_bmi, log_sigma_kcal])))

    ### --- Identify dimensions of coefficient vector
    # p INCLUDES INTERCEPT 
    p = beta.shape[0]
    n = X.shape[0]
    X_true_with_intercept = jnp.concatenate([jnp.ones((n, 1)), X_true], axis=1)
    eta = X_true_with_intercept @ beta

    # ! Density is expressed in terms of log_sigma seen from last summand in log_sigma_prior_term
    log_likelihood_term = - n / 2 * log_sigma - n / 2 * (log_sigma_age + log_sigma_bmi + log_sigma_kcal) - 1 / 2 * (y - eta) @ (y - eta) / jnp.exp(log_sigma) 
    # Pretty odd to use the design matrix, not a sum. But I think, a sum would be less efficient: 
        # 1) Build difference and square each entry in the matrix (n x 3) bzw. (n x p-1) where p is number covariates INCLUDING the intercept
        # 2) Square each entry in the matrix for squared difference
        # 3) Multiply each entry by the inverse of the error variance for the respective covariate (i.e. the respective entry on the diagonal of the covariance matrix)
        # 4) Sum across covariates (i.e. across columns) to get a vector of length n, where each entry is the sum of the error terms across covariates for the respective row; this is the vector of measurement error terms for each row
    log_measurement_error_term = - 1 / 2 * (X_no_intercept - X_true)**2 @ jnp.linalg.inv(error_cov_matrix).sum(axis = 1)
    log_beta_prior_term = - p / 2 * jnp.log(b ** 2) - 1 / (2 * b ** 2) * jnp.sum(beta ** 2)
    log_sigma_prior_term = (c - 1) * log_sigma - d * jnp.exp(log_sigma) + log_sigma
    log_sigma_error_term = (e - 1) * (log_sigma_age + log_sigma_bmi + log_sigma_kcal) - f * (jnp.exp(log_sigma_age) + jnp.exp(log_sigma_bmi) + jnp.exp(log_sigma_kcal)) + log_sigma_age + log_sigma_bmi + log_sigma_kcal
    log_empirical_density = empirical_logdensity(X_true).sum()
    return log_likelihood_term + log_measurement_error_term.sum() + log_empirical_density + log_beta_prior_term + log_sigma_prior_term + log_sigma_error_term




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
        cols_excluded_from_error=cols_excluded_from_error,
        seed = seed
    )
    return voe.masked_data[cols]

def quantify_bias(
        raw_data, 
        error_type, error_factor, cols_excluded_from_error,
        log_corrected_density, rng_key, 
        empirical_logdensity,
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
        # naive = BHM(
        #     data = data, 
        #     response = "LBXT4",
        #     covariates = ["RIDAGEYR", "bmi", "DR1TKCAL"],
        #     post_log_dens = post_log_dens,
        #     hyperparams = {
        #         "b": 100, # TODO: Depends on scale, of course. e.g. age reaches up to 80, calories reach up to 4k, bmi reaches up to 40; this is the SD of the posterior
        #         "c": 1,
        #         "d": 1
        #     },
        #     initial_positions = {
        #         "beta": jnp.zeros((num_chains, p)),
        #         "log_sigma": jnp.ones((num_chains, ))
        #     }, 
        #     empirical_logdensity = dummy_empirical_logdensity,
        #     inverse_mass_matrix = jnp.eye(3),
        #     rng_key = rng_key,
        #     num_chains = num_chains,
        #     inital_step_size = 1e-3
        # )
        # naive.fit()
        # # Bias is then the deviation of the naive model's estimates from the true parameters (or frequentistic OLS estimates)
        # naive_bias_estimate = naive.mean_estimates(param_name = "beta") - bhm.mean_estimates(param_name = "beta")
        # bias_estimates["naive"].append(naive_bias_estimate)


        corrected = BHM(
            data = data, 
            response = "LBXT4",
            covariates = ["RIDAGEYR", "bmi", "DR1TKCAL"],
            post_log_dens = log_corrected_density,
            hyperparams = {
                "b": 100, # TODO: Depends on scale, of course. e.g. age reaches up to 80, calories reach up to 4k, bmi reaches up to 40; this is the SD of the posterior
                "c": 1,
                "d": 1,
                "e": 1,
                "f": 1
            },
            empirical_logdensity = empirical_logdensity,
            initial_positions = {
                "beta": jnp.zeros((num_chains, p)),
                "log_sigma": jnp.ones((num_chains, )), 
                "log_sigma_age": jnp.ones((num_chains, )),
                "log_sigma_bmi": jnp.ones((num_chains, )),
                "log_sigma_kcal": jnp.ones((num_chains, )),
                # This is a bit of a hack: 
                # I need to pass the true covariate values to the log density function to express the measurement error density; 
                # this is how I do it for now: Pass observed values in, 
                # but it might be better to change the structure of the BHM class to allow for passing additional data to the log density function in a more elegant way
                "X_true": jnp.tile(data[["RIDAGEYR", "bmi", "DR1TKCAL"]].values, (num_chains, 1, 1)) 
            }, 
            inverse_mass_matrix = jnp.eye(3),
            rng_key = rng_key,
            num_chains = num_chains,
            inital_step_size = 1e-3, 
            burnin = 5000,
            warmup_steps=5000, 
            n_samples=10000
        )
        corrected.fit()
        # Bias is then the deviation of the corrected model's estimates from the true parameters (or frequentistic OLS estimates)
        corrected_bias_estimate = corrected.mean_estimates(param_name = "beta") - bhm.mean_estimates(param_name = "beta")
        bias_estimates["corrected"].append(corrected_bias_estimate)
    return corrected
    return bias_estimates

test = quantify_bias(
    raw_data = voe_data.dropna(ignore_index = True), 
    error_type = "normal", error_factor = normal_sd, cols_excluded_from_error = None,
    log_corrected_density = post_log_dens_gaussian_additive,
    B = 1, rng_key = rng_key, empirical_logdensity =  lazy_empirical_logdensity
)












# %%
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