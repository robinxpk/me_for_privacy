# %%
import jax
import jax.numpy as jnp
import jax.nn as jnn
import blackjax
import matplotlib.pyplot as plt
import numpy as np
from ME.Data import Data
import pandas as pd

### Draw rng_key for later use
from datetime import date
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))

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
    error_factors = np.array([normal_sd]), 
    error_type="normal"
)
# %%
X = voe_normal.raw_data[["RIDAGEYR", "DR1TKCAL", "bmi"]].values
X = jnp.c_[jnp.ones(X.shape[0])[:, None], X]
y = voe_normal.raw_data["LBXT4"].values
#%%&
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
# 1) Building the kernel
p = X.shape[1] - 1
step_size = 1e-9
inverse_mass_matrix = jnp.array([0.01, 0.01, 0.01, 0.01, 0.01])
### --- blackjax.nuts expetcts a logdensity function that only depends on the parameter-values
logdensity_fn = lambda params: post_log_dens(y, X, params, b = 10 ** 3, c = 1, d = 1)
nuts = blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix)
# %%
# 2) Initialize the state 
initial_position = {
    "beta": jnp.array([8.7, 0.002, -0.0004, 0.075]),
    "log_sigma": jnp.log(jnp.std(y))
}
initial_state = nuts.init(initial_position)
state = nuts.init(initial_position)
# %%
# 3) Iterate
rng_key, init_key = jax.random.split(rng_key)

### Define function to take steps / draw samples
def inference_loop(rng_key, kernel, initial_state, num_samples): 
    ### --- Use just in time compilation to improve runtime
    @jax.jit
    def one_step(state, rng_key): 
        new_state, info = kernel(rng_key, state)
        ### --- jax.lax.scan expects two outputs: a carry (first) and an output (second).
        ## 1. carry: Updated chain state that is input for next step.
        ## 2. (per-iteration) output: Output that the scan-function stacks into the states object which is returned at the end.
        return new_state, (new_state, info)
    
    keys = jax.random.split(rng_key, num_samples)
    final_state, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
    
    return final_state, states, infos

### --- Run the Markov Chain
rng_key, sample_key = jax.random.split(rng_key)
final_state, states, infos = inference_loop(sample_key, nuts.step, initial_state, num_samples = 5_000)
# %%
# 4) Visualize chain
### --- Burn-in size
burnin = 1_000
### --- Plot traces
## -- Beta coefficients
fig, ax = plt.subplots(1, 4, figsize=(12, 2))
for i, axi in enumerate(ax):
    axi.plot(states.position["beta"][:, i])
    axi.set_title(f"$\\beta_{i}$")
    axi.axvline(x=burnin, c="tab:red")
plt.show()


# %%
## -- Variance(beta) - log scale
plt.plot(states.position["log_sigma"])
plt.title("$\\log \\sigma^2$")
plt.axvline(x=burnin, c="tab:red")
plt.show()
## -- Variance(beta) - variance scale
plt.plot(jnp.exp(states.position["log_sigma"]))
plt.title("$\\sigma^2$")
plt.axvline(x=burnin, c="tab:red")
plt.show()
# %%
