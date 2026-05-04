# %%
import jax.numpy as jnp
import pandas as pd
import jax
import pytest

from jax.scipy.stats import gaussian_kde
from ME.functions import post_log_dens, post_log_dens_gaussian_additive, post_log_dens_lognormal_multiplicative, post_log_dens_epit
from ME.BHM import BHM
from ME.Data import Data
from ME.KDE import KDE_Dummy_Model


def test_debug_epit_posterior(): 
    # This test uses the sigmoid fit to the DR1TKCAL variable 
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

    variable_subset = ["LBXT4", "RIDAGEYR", "bmi", "DR1TKCAL"]
    response = "LBXT4"
    covariates = ["RIDAGEYR", "bmi", "DR1TKCAL"]
    error_cols = ["DR1TKCAL"]
    # Setup Data
    voe_data = pd.read_csv(r"../data/voe_data.csv", sep = ";", header = 0)

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
        raw_data = voe_data.dropna(ignore_index = True).loc[:, variable_subset],
        error_type = "none"
    )

    error_var = 0.3
    voe_error= Data(
        name = f"epit_{error_var}", 
        raw_data = voe_data.dropna(ignore_index = True), 
        seed = 1234,
        error_vars = {"DR1TKCAL": jnp.array([error_var])}, 
        error_type="ePIT", 
        # Exclude the error on age and bmi for now to simplify the error structure
        cols_excluded_from_error = ["LBXT4", "RIDAGEYR", "bmi"],
        e_sigmoid = e_sigmoid, 
        e_inv_sigmoid = e_inv_sigmoid
    )

    # For the error column, we habe tilde(x) as of now. 
    # The BHM model (as of now) expected tilde(z) values where columns are touched by error
    # Thus, replace tilde(x) with tilde(z) in the design matrix
        # TODO: IMO this is not intuitive when using this error. Might adjust such that its always on x scale and invert  within the BHM object (setter and getter?)
    voe_error.masked_data.loc[:, "DR1TKCAL"] = jax.scipy.stats.norm.ppf(e_sigmoid(voe_error.masked_data.loc[:, "DR1TKCAL"].values)) 

    # Because I draw z, the KDE must be expressed in terms of Z, too. 
    # Basically, the design matrix is X, but I evaluate partially X and z where touched by error

    data_raw_with_z = voe.raw_data
    data_raw_with_z.loc[:, "DR1TKCAL"] = jax.scipy.stats.norm.ppf(e_sigmoid(voe_error.raw_data.loc[:, "DR1TKCAL"].values)) 
    empirical_kde_mdl = gaussian_kde(data_raw_with_z.loc[:, covariates].values.T, bw_method = "scott")

    num_chains = 1
    error_cov_matrix = jnp.diag(jnp.array([error_var])) 
    hyperparams = {
        "b": 100, 
        "c": 1,
        "d": 1
    }
    params= {
        "beta": jnp.repeat(
            jnp.array([0, 0, 0, 0]), 
            repeats = num_chains,
            axis = 0
        ),
        "log_sigma": jnp.repeat(jnp.log(jnp.var(jnp.asarray(voe_error.raw_data.loc[:, response].values))), num_chains, axis = 0), 
        "Z_true": jnp.tile(data_raw_with_z.loc[:, ["DR1TKCAL"]].values, (num_chains, 1, 1)) 
    }
    error_cols_index = [covariates.index(col) for col in error_cols] 

    X = voe_error.masked_data.loc[:, covariates]

    logdensity_fn = lambda params: post_log_dens_epit(
        jnp.array(voe_error.masked_data.loc[:, response].values), 
        jnp.c_[jnp.ones(X.shape[0])[:, None], X], 
        params, 
        error_cols_index, 
        error_cov_matrix, 
        empirical_kde_mdl, 
        e_sigmoid, e_inv_sigmoid, 
        **hyperparams
    )

    logdensity_fn(params)


test_debug_epit_posterior()

# %%
