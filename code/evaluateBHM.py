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
data_path = r"../data/"

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
normal_sd = 10 ** 3 # NOTE: Depends on scale, of course. e.g. calories reach up to 4k
voe_normal= Data(
    name = f"normal_{normal_sd}", 
    raw_data = voe_data.dropna(ignore_index = True), 
    prob = 1, 
    error_factors = jnp.array([normal_sd]), 
    error_type="normal"
)


data = voe_normal.raw_data[["LBXT4", "RIDAGEYR", "DR1TKCAL", "bmi"]]
# -1 for response variable, +1 for intercept; This is only kept for clarity
p = data.shape[1] - 1 + 1
num_chains = 4

# %%
bhm = BHM(
    data = data, 
    response = "LBXT4",
    covariates = ["RIDAGEYR", "DR1TKCAL", "bmi"],
    post_log_dens = post_log_dens,
    hyperparams = {
        "b": 10,
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
fig, ax = plt.subplots(1, 4, figsize=(12, 2))
for i, axi in enumerate(ax):
    axi.plot(bhm.res.position["beta"][1][:, i])
    axi.plot(bhm.res.position["beta"][2][:, i])
    axi.plot(bhm.res.position["beta"][3][:, i])
    axi.plot(bhm.res.position["beta"][4][:, i])
    axi.set_title(f"$\\beta_{i}$")
plt.show()

# %%
## -- Variance(beta) - log scale
plt.plot(bhm.res.position["log_sigma"][1])
plt.plot(bhm.res.position["log_sigma"][2])
plt.plot(bhm.res.position["log_sigma"][3])
plt.plot(bhm.res.position["log_sigma"][4])
plt.title("$\\log \\sigma^2$")
plt.show()
## -- Variance(beta) - variance scale
plt.plot(jnp.exp(bhm.res.position["log_sigma"][1]))
plt.plot(jnp.exp(bhm.res.position["log_sigma"][2]))
plt.plot(jnp.exp(bhm.res.position["log_sigma"][3]))
plt.plot(jnp.exp(bhm.res.position["log_sigma"][4]))
plt.title("$\\sigma^2$")
plt.show()
# %%
