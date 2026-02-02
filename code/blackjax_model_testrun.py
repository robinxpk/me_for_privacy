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

# 0) Ensure variable types
y = y.astype(jnp.float32)
p = X.shape[1]

# 1) Building the kernel
step_size = 1e-3
inverse_mass_matrix = jnp.ones(p + 2)
### --- blackjax.nuts expetcts a logdensity function that only depends on the parameter-values
logdensity_fn = lambda params: post_log_dens(y, X, params)
nuts = blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix)

# 2) Initialize the state 
initial_position = {
    "beta": jnp.ones(p),
    "log_sigma_sq_beta": 1., 
    "mean_beta": 1.
}
initial_state = nuts.init(initial_position)
state = nuts.init(initial_position)

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

# 4) Visualize chain

### --- Burn-in size
burnin = 1_000
### --- Plot traces
## -- Beta coefficients
fig, ax = plt.subplots(1, 3, figsize=(12, 2))
for i, axi in enumerate(ax):
    axi.plot(states.position["beta"][:, i])
    axi.set_title(f"$\\beta_{i}$")
    axi.axvline(x=burnin, c="tab:red")
plt.show()
## -- Mean beta
plt.plot(states.position["mean_beta"])
plt.title("$\\mu_{\\beta}$")
plt.axvline(x=burnin, c="tab:red")
plt.show()
## -- Variance(beta) - log scale
plt.plot(states.position["log_sigma_sq_beta"])
plt.title("$\\log \\sigma^2_{\\beta}$")
plt.axvline(x=burnin, c="tab:red")
plt.show()
## -- Variance(beta) - variance scale
plt.plot(jnp.exp(states.position["log_sigma_sq_beta"]))
plt.title("$\\sigma^2_{\\beta}$")
plt.axvline(x=burnin, c="tab:red")
plt.show()

# %%
# 5) Predictive distribution
chains = states.position["beta"][burnin:, :]
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
