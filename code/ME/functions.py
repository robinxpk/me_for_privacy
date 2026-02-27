import jax.numpy as jnp
import jax
# This file contains the functions used the in main script. 


### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###
#!! ----------------- Posterior log Density Definitions ------------------------------- !!#
### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###

# 0 - NAIVE) Posterior log Density for model WITHOUT measurement error / the NAIVE model
def post_log_dens(y, X, params, error_cols, error_cov_matrix, empirical_kde_mdl, b, c, d):
    ### --- params is a dictionary containing the current value of the parameters
    ### --- Assign dictionary entries to variables 
    beta = params["beta"]
    log_sigma = params["log_sigma"]
    sigma = jnp.exp(log_sigma)
    ### --- Identify dimensions of coefficient vector
    n = X.shape[0]
    eta = X @ beta
    # --> Error informations are not used (i.e. error_cols and error_cov_matrix). 

    # ! Density is expressed in terms of log_sigma seen from last summand in log_sigma_prior_term
    log_likelihood_term = - n / 2 * jnp.log(sigma ** 2) - 1 / (2 * sigma ** 2) * jnp.sum((y - eta) ** 2)
    log_beta_prior_term = - 1 / (2 * b ** 2) * jnp.sum(beta ** 2)
    log_sigma_prior_term = (c - 1) * jnp.log(sigma ** 2) - d * sigma ** 2 + jnp.log(sigma ** 2) 
    log_empirical_density = empirical_kde_mdl.logpdf(X).sum()
    return log_likelihood_term.sum() + log_empirical_density + log_beta_prior_term + log_sigma_prior_term 


# 1 - ADD GAUSSIAN) Posterior log Density for ADDITIVE GAUSSIAN error
def post_log_dens_gaussian_additive(y, X, params, error_cols, error_cov_matrix, mdl, b, c, d,):
    # ! The error_cov_matrix is assumed to be a diagonal matrix! The density below uses simplifications based on this assumption
    ### --- params is a dictionary containing the current value of the parameters
    ### --- Assign dictionary entries to variables 
    beta = params["beta"]
    log_sigma = params["log_sigma"]
    # To extract the true observed values, I need the design matrix only containing actual covariates, i.e. without intercept
    X_no_intercept = X[:, 1:] 
    X_error_cols = X_no_intercept[:, error_cols]
    # Construct X_true based on params["X_true"] and the columns without error: 
    # i.e. replace the columns in the design matrix which are touched by error using the sampled covariates
    X_true = X_no_intercept.at[:, error_cols].set(params["X_true"])
    X_true_error_cols = X_true[:, error_cols]

    ### --- Identify dimensions of coefficient vector
    # p INCLUDES INTERCEPT 
    p = beta.shape[0]
    n = X.shape[0]
    X_true_with_intercept = jnp.concatenate([jnp.ones((n, 1)), X_true], axis=1)
    eta = X_true_with_intercept @ beta

    # ! Density is expressed in terms of log_sigma seen from last summand in log_sigma_prior_term
    log_likelihood_term = - n / 2 * log_sigma  - 1 / 2 * (y - eta) @ (y - eta) / jnp.exp(log_sigma) 
    # Pretty odd to use the design matrix, not a sum. But I think, a sum would be less efficient: 
        # 1) Build difference and square each entry in the matrix (n x 3) bzw. (n x p-1) where p is number covariates INCLUDING the intercept
        # 2) Square each entry in the matrix for squared difference
        # 3) Multiply each entry by the inverse of the error variance for the respective covariate (i.e. the respective entry on the diagonal of the covariance matrix)
        # 4) Sum across covariates (i.e. across columns) to get a vector of length n, where each entry is the sum of the error terms across covariates for the respective row; this is the vector of measurement error terms for each row
    log_measurement_error_term = - 1 / 2 * (X_error_cols - X_true_error_cols)**2 @ jnp.linalg.inv(error_cov_matrix).sum(axis = 1)
    log_beta_prior_term = - p / 2 * jnp.log(b ** 2) - 1 / (2 * b ** 2) * jnp.sum(beta ** 2)
    log_sigma_prior_term = (c - 1) * log_sigma - d * jnp.exp(log_sigma) + log_sigma
    log_empirical_density = mdl.logpdf(X_true.T).sum()
    ## Print statement I used to check if the empirical density actually varied. It does :) 
    return log_likelihood_term + log_measurement_error_term.sum() + log_empirical_density + log_beta_prior_term + log_sigma_prior_term 