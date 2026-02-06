# %%
import jax
import jax.numpy as jnp
import jax.nn as jnn
import blackjax
import matplotlib.pyplot as plt

### Draw rng_key for later use
from datetime import date
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))

### Generate data
from sklearn.datasets import make_biclusters
num_points = 50

X, rows, cols = make_biclusters(
    (num_points, 2), 2, noise=0.6, random_state=314, minval=-3, maxval=3
)
y = rows[0] * 1.0  # y[i] = whether point i belongs to cluster 1

colors = ["tab:red" if el else "tab:blue" for el in rows[0]]
plt.scatter(*X.T, edgecolors=colors, c="none")
plt.xlabel(r"$X_0$")
plt.ylabel(r"$X_1$")
# Add columns with all ones for intercept to design matrix 
X = jnp.c_[jnp.ones(num_points)[:, None], X]


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
# 0) Ensure variable types
y = y.astype(jnp.float32)
p = X.shape[1]

# %%
# Assume this function has been specified above.
logdensity_fn = lambda params: post_log_dens(y, X, params)

# Set parameters
num_chains = 4
step_size = 1e-3
inverse_mass_matrix = jnp.ones(p + 2)
nuts = blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix)
initial_positions = {
    "beta": jnp.ones((num_chains, p)),
    "log_sigma_sq_beta": jnp.ones((num_chains, )) , 
    "mean_beta": jnp.ones((num_chains, ))
}

# %% 
### Create a windows adaptation object to repeatedly call it for each chain. 
warmup = blackjax.window_adaptation(
    blackjax.nuts,
    logdensity_fn
)

# %%
### --- Inference Loop
def inference_loop(rng_key, kernel, initial_state, num_samples):

    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states

# %%
def adapt_and_sample_one_chain(rng_key, initial_position):
    
    ## Key handling
    adapt_key, sample_key = jax.random.split(rng_key)
    
    ## For a given initial position, get 
    (state, parameters), info = warmup.run(
        adapt_key,
        initial_position,
        num_steps = 1_000
    )
    nuts = blackjax.nuts(logdensity_fn, **parameters)

    chain = inference_loop(
        sample_key, 
        nuts.step, 
        state, 
        num_samples = 5_000
    )

    return chain
# %%
### --- Run the Markov Chain
rng_key = jax.random.split(rng_key, num_chains)
res = jax.vmap(adapt_and_sample_one_chain)(rng_key, initial_positions)

# %%
### --- Burn-in size
burnin = 1_000

### --- Plot traces
## -- Beta coefficients
fig, ax = plt.subplots(1, 3, figsize=(12, 2))
for i, axi in enumerate(ax):
    axi.plot(res.position["beta"][1][:, i], alpha = 0.5)
    axi.plot(res.position["beta"][2][:, i], alpha = 0.5)
    axi.plot(res.position["beta"][3][:, i], alpha = 0.5)
    axi.plot(res.position["beta"][4][:, i], alpha = 0.5)
    axi.set_title(f"$\\beta_{i}$")
    axi.axvline(x=burnin, c="tab:red")
plt.show()
## -- Mean beta
plt.plot(res.position["mean_beta"][1])
plt.plot(res.position["mean_beta"][2])
plt.plot(res.position["mean_beta"][3])
plt.plot(res.position["mean_beta"][4])
plt.title("$\\mu_{\\beta}$")
plt.axvline(x=burnin, c="tab:red")
plt.show()
## -- Variance(beta) - log scale
plt.plot(res.position["log_sigma_sq_beta"][1])
plt.plot(res.position["log_sigma_sq_beta"][2])
plt.plot(res.position["log_sigma_sq_beta"][3])
plt.plot(res.position["log_sigma_sq_beta"][4])
plt.title("$\\log \\sigma^2_{\\beta}$")
plt.axvline(x=burnin, c="tab:red")
plt.show()
## -- Variance(beta) - variance scale
plt.plot(jnp.exp(res.position["log_sigma_sq_beta"][1]))
plt.plot(jnp.exp(res.position["log_sigma_sq_beta"][2]))
plt.plot(jnp.exp(res.position["log_sigma_sq_beta"][3]))
plt.plot(jnp.exp(res.position["log_sigma_sq_beta"][4]))
plt.title("$\\sigma^2_{\\beta}$")
plt.axvline(x=burnin, c="tab:red")
plt.show()

# %%
# 5) Predictive distribution
chain = 1
chains = res.position["beta"][chain][burnin:, :]
nsamp, _ = chains.shape
# Create a meshgrid
X_no_intercep = X[:, 1:3]  # Remove intercept for plotting
xmin, ymin = X_no_intercep.min(axis=0) - 0.1
xmax, ymax = X_no_intercep.max(axis=0) + 0.1
step = 0.1
Xspace = jnp.mgrid[xmin:xmax:step, ymin:ymax:step]
_, nx, ny = Xspace.shape

# Compute the average probability to belong to the first cluster at each point on the meshgrid
Phispace = jnp.concatenate([jnp.ones((1, nx, ny)), Xspace])
Z_mcmc = jnn.sigmoid(jnp.einsum("mij,sm->sij", Phispace, chains))
Z_mcmc = Z_mcmc.mean(axis=0)

plt.contourf(*Xspace, Z_mcmc)
plt.scatter(*X_no_intercep.T, c=colors)
plt.xlabel(r"$X_0$")
plt.ylabel(r"$X_1$")
# %%
