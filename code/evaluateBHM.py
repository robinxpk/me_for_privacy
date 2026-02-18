# %%
from ME.BHM import BHM
import jax
import jax.numpy as jnp
import jax.nn as jnn
import blackjax
import matplotlib.pyplot as plt
import pandas as pd


from datetime import date
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))

def post_log_dens(y, X, params):
    ### --- params is a dictionary containing the current value of the parameters
    ### --- Assign dictionary entries to variables 
    beta = params["beta"]
    mean_beta = params["mean_beta"] 
    log_sigma_sq_beta = params["log_sigma_sq_beta"]
    ### --- Identify dimensions of coefficient vector
    p = beta.shape[0]
    eta = X @ beta

    log_likelihood_term = y * jnn.log_sigmoid(eta) + (1 - y) * jnn.log_sigmoid(-eta)
    beta_prior_term_1 = - p / 2 * log_sigma_sq_beta 
    beta_prior_term_2 = - 1 / 2 * jnp.exp(- log_sigma_sq_beta) * (beta - mean_beta) @ (beta - mean_beta)
    beta_hyperpriors = - mean_beta ** 2 / (2 * 100 ** 2) - jnp.exp(log_sigma_sq_beta) + log_sigma_sq_beta
    return log_likelihood_term.sum() + beta_prior_term_1 + beta_prior_term_2 + beta_hyperpriors

# %%
### Draw rng_key for later use

### Generate data
from sklearn.datasets import make_biclusters
num_points = 50

X, rows, cols = make_biclusters(
    (num_points, 2), 2, noise=0.6, random_state=314, minval=-3, maxval=3
)
y = rows[0] * 1.0  # y[i] = whether point i belongs to cluster 1
data = pd.concat([pd.DataFrame(X, columns=["X0", "X1"]), pd.DataFrame(y, columns=["y"])], axis=1)

colors = ["tab:red" if el else "tab:blue" for el in rows[0]]
plt.scatter(*X.T, edgecolors=colors, c="none")
plt.xlabel(r"$X_0$")
plt.ylabel(r"$X_1$")
# %%
num_chains = 4
p = X.shape[1] + 1

bhm = BHM(
    data = data, 
    response = "y",
    covariates = ["X0", "X1"],
    post_log_dens = post_log_dens,
    initial_positions = {
        "beta": jnp.ones((num_chains, p)),
        "log_sigma_sq_beta": jnp.ones((num_chains, )) , 
        "mean_beta": jnp.ones((num_chains, ))
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
