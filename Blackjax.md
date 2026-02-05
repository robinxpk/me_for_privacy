#### TODOs ####
- [ ] Run example, but increase number of chains.
---

See also [[Jax]] and [NumPyro Tutorial](https://num.pyro.ai/en/stable/tutorials/). 

[Official Blackjax Page](https://blackjax-devs.github.io/blackjax/)


## Logistic Regression ##
[Tutorial on this](https://blackjax-devs.github.io/sampling-book/models/logistic_regression.html)

#### Hierarchical Model ####

1. Likelihood
$$
\begin{align}
    y_i\mid \boldsymbol{x}_i, \boldsymbol{\beta} \sim Bernoulli(\pi_i) \\
    \log\left(  \frac{\pi_i}{1-\pi_i}\right) = \boldsymbol{x}^T_i\boldsymbol{\beta}\\
\end{align}
$$ 
for $i = 1, ..., n$ (-- Likelihood not conditioned on $\pi$, because redundant if $\boldsymbol{x}_i$ and $\boldsymbol{\beta}$ given). 

2. Priors
$$
\begin{align}
    \beta_j\mid \mu_\beta, \sigma^2_{\beta} \overset{iid.}{\sim} N(\mu_\beta, \sigma^2_{\beta})
\end{align}
$$
for $j = 1, ..., p$. 

3. Hyperpriors
$$
\begin{align}
    \mu_{\beta}&\sim N(0, 100^2 ) \\
    \sigma_{\beta}^2 &\sim Ga(\text{shape} = 1, \text{rate}= 1)
\end{align}
$$

The Likelihood bzw. the distribution of $y_i|\boldsymbol{\beta}$ is not conditional on $\pi_i$. 
Because given the covariates $\boldsymbol{x}_i$ and $\boldsymbol{\beta}$, the probability $\pi_i$ is given by the logit-link and the data distribution is fully specified if $\boldsymbol{\beta}$ is given. </br>
To emphasise this, one may also write: $y_i\mid \boldsymbol{\beta} \sim Bernoulli(logit^{-1}(\boldsymbol{x}_i^T\boldsymbol{\beta}))$.

This setup implies that one must not include a prior for $\pi$ itself! This would only be the case if we were to not model $\pi$ using covariates. 

The unnormalized  joint posterior: 
$$
\begin{align}
    p(\boldsymbol{\beta}, \mu_\beta, \sigma^2_\beta \mid \boldsymbol{y}, \boldsymbol{X}) 
    \propto 
    \left( \prod_{i=1}^{n}{p(y_i\mid \boldsymbol{x}_i, \boldsymbol{\beta})} \right) 
    \left(\prod_{j=1}^{p}{p(\beta_j\mid \mu_\beta, \sigma^2_{\beta})} \right) 
    p(\mu_{\beta}) p(\sigma^2_{\beta}). 
\end{align}
$$
Taking the $\log$ and plugging in the distributions, the unnormalized $\log$-posterior is defined as
$$
\begin{align*}
    \log p(\boldsymbol{\beta}, \mu_\beta, \sigma^2_\beta \mid \boldsymbol{y}, \boldsymbol{X}) 
    &= 
    %%% --- Likelihood
    \left( \sum_{i=1}^{n}{y_i \log\left[ logit^{-1}(\boldsymbol{x}_i^T \boldsymbol{\beta}) \right] + (1-y_i) \log\left[ 1-logit^{-1}(\boldsymbol{x}^T_i \boldsymbol{\beta})\right] } \right) \\
    %%% --- Beta
    & + \left( -\frac{p}{2}\log(\sigma^2_{\beta}) - \frac{1}{2\sigma^2_\beta}\sum_{j=1}^{p}{(\beta_j - \mu_{\beta})^2} \right) \\
    %%% --- Hyperpriors
    & - \frac{\mu^2_\beta}{2 \cdot 100^2} - \sigma^2_{\beta}
    + C.
\end{align*}
$$

**! Important**: NUTS (the sampler I'll use) samples from the real line! In theory, I could just reject proposals that are negative, but this is not implemented in libraries like BlackJax and it seems wasteful on a computational / time level. Also, implementation seems odd since HMC bzw. NUTS relies on numerically integrating Hamiltonian dynamics (Not sure what that means). </br>
Thus, variances (std. Deviations) should be transformed using the log! Due to this, we run into *change-of-variable* (see @gelmanBDA, p. 21): 

**Change / Transformation of variable:** If $p_u$ is continuous distribution and $v = f(u)$ is a one-to-one transformation, then the joint density of the transformed variable is 
$$
\begin{align}
    p_{v}(v) = |J|p_{u}(f^{-1}(v)). 
\end{align}
$$
Where $|J|$ is the absolute value of the determinant of the Jacobian of the transformation $f^{-1}(v)$: $J = \frac{\delta f^{-1}(v)}{\delta v}$.
i.e. A density **is *not* invariant** to reparametrization. Basically, this means that if I sample from space $\log(\sigma_{\beta}^2)$, I need to express the posterior in terms of **this** (prior-)density. Else, I evaluate the posterior in a non-equivalent way caused by not accounting for the transformation in the (prior-)density: </br>
Right now, the above posterior is expressed in terms of $\sigma^2_{\beta}$ which I would like to transform into $\log(\sigma^2_{\beta})$.
Apply change of variable for $u := \sigma^2_\beta$ and $v:=\log(\sigma^2_\beta)$ where $u$ is the random variable I start with and $v$ is the transformed variable and $u := f^{-1}(v)$. This implies $v = f(u) = \log(u)$ and $u = f^{-1}(v) = \exp(v)$. The (determinant of the) Jacobian is then
$$
\begin{align}
    |J|= \frac{\delta u}{\delta v} = \frac{\delta f^{-1}(v)}{\delta v}  = \frac{\delta \exp(v)}{\delta v} = \exp(v). 
\end{align}
$$ 
Plugged the variance in, we obtain
$$
\begin{align}
    p_{v}\left(v\right) &= \exp(v)\cdot p_{u}\left(\exp(v)\right) \ \ \mid v = \log(\sigma^2_{\beta})\\
    % p_{v}\left(v = \log(\sigma_{\beta}^2)\right) &= \exp(\log(\sigma_{\beta}^2))\cdot p_{u}\left(\exp(\log(\sigma^2_\beta))\right). 
    \leftrightarrow \log(p_v(v)) &= v + \log\left(p_{u}(\exp(v)\right).
\end{align}
$$
Using this transformation, we can express the posterior in terms of $v = \log(\sigma^2_{\beta})$. 

Note: 
$$
\begin{align}
    p(\boldsymbol{\beta}, \mu_\beta, \color{red}\sigma^2_\beta \color{default}\mid \boldsymbol{y}) \color{red}\neq\color{default}
    p(\boldsymbol{\beta}, \mu_\beta, \color{red}\log(\sigma^2_\beta)\color{default} \mid \boldsymbol{y}), 
\end{align}
$$
because these are different joint densities! These are *only* equivalent if $|J| = 1 \leftrightarrow p_{v} = p_{u}$. 
However, what **does** remain constant is the "probability content". That is, for any set $A \subset (0, \infty)$
$$
\begin{align}
    \mathbb{P}(\sigma^2_{\beta} \in A) = \mathbb{P}(\log(\sigma^2_{\beta}) \in \log(A))
\end{align}
$$
Intuitively, this means that the two random variable describe the same underlying uncertainty, but on different scales.

Adjust the posterior log-density accordingly 
$$
\begin{align}
    \log p(\boldsymbol{\beta}, \mu_\beta, v \mid \boldsymbol{y}, \boldsymbol{X}) 
    & = 
    %%% --- Likelihood
    \left( \sum_{i=1}^{n}{y_i \log\left[ logit^{-1}(\boldsymbol{x}_i^T \boldsymbol{\beta}) \right] + (1-y_i) \log\left[ 1-logit^{-1}(\boldsymbol{x}^T_i \boldsymbol{\beta})\right] } \right) \\
    %%% --- Beta
    & + \left( -\frac{p}{2}\underbrace{v}_{\text{Replace }\log(\sigma^2_\beta)} - \frac{1}{2}\frac{1}{\underbrace{\exp(v)}_{\text{Replace } 1 / \sigma^2_{\beta}}}\sum_{j=1}^{p}{(\beta_j - \mu_{\beta})^2} \right) \\
    %%% --- Hyperpriors
    & - \frac{\mu^2_\beta}{2 \cdot 100^2} + \underbrace{v - \exp(v)}_{\log(p^{(Gamma)}_{v}(v))}
    + C.
\end{align}
$$

#### Implementation ####
Setup
```
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
```

Then, specify the posterior log-density and helper functions 
```
def inv_logit(eta):
    # logit^{-1}(eta)
    return jnp.exp(eta) / (1 + jnp.exp(eta))

def log_inv_logit(eta): 
    # log( logit^{-1}(eta) )
    return -jnp.log(1 + jnp.exp(- eta))

def post_log_dens(y, X, beta, sigma_sq_beta, mean_beta):
    eta = X @ beta
    p = beta.shape[0]

    log_likelihood_term = y * log_inv_logit(eta) + (1 - y) * jnp.log((1 - inv_logit(eta)))
    beta_prior_term_1 = - p / 2 * jnp.log(sigma_sq_beta) 
    beta_prior_term_2 = - 1 / (2 * sigma_sq_beta) * (beta - mean_beta) @ (beta - mean_beta)
    beta_hyperpriors = - mean_beta ** 2 / (2 * 100 ** 2) - sigma_sq_beta
    return log_likelihood_term.sum() + beta_prior_term_1 + beta_prior_term_2 + beta_hyperpriors
```
This is a raw implementation that should work. However, if a function already exists in [[Jax]], utilize it! It is less prone to errors and usually numerically more stable. Here, the [log-sigmoid function](https://docs.jax.dev/en/latest/_autosummary/jax.nn.log_sigmoid.html) is already implemented in Jax as `jnn.log_sigmoid`. See also [sigmoid function](https://docs.jax.dev/en/latest/_autosummary/jax.nn.log_sigmoid.html) `jnn.sigmoid`.  

Also, because the following uses the NUTS, adjust the log posterior density to be defined for draws from $\log(\sigma^2_{\beta})$.

Implementation using Jax-functions and change of variable: 
```
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
```
*Note*: Instead of the variance $\sigma^2_\beta$, I should rather use the standard deviation: 
- Numerically more stable. 
- Curvature and funnels (Note sure what this is). 

But I skip this for now, guess it will work with the variance, too. 

With this, we can start sampling using the No U-Turn Sampler (NUTS): 
```
# 0) Ensure variable types
y = y.astype(jnp.float32)
p = X.shape[1]

# 1) Building the kernel
step_size = 1e-3
inverse_mass_matrix = jnp.ones(p + 2)
### --- blackjax.nuts expetcts a logdensity function that only depends on the parameter-values
logdensity_fn = lambda params: post_log_dens(y, X, params)
nuts = blackjax.nuts(logdensity_fn, step_size, inverse_mass_matrix)
```
`blockjax.nuts(.)` builds a `SamplingAlgorithm`-object with (at least) two callables: 
- `nuts.init(position)` to create initial state.
- `nuts.step(rng_key, state)` to do **one** Markov transition.

The `step` function is also called *kernel*: A Function that takes a current state (and randomness) and returns a new state plus diagnostics is a transition kernel; See the *How does it work?*-paragraph on [Blackjax Github](https://github.com/blackjax-devs/blackjax). 
It is implemented as `new_state, info = kernel(rng_key, state)`. The NUTS-step is described [here](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/mcmc/nuts/index.html).

NUTS (and Hamiltonian MC [HMC]) use an auxiliary momentum variable and define a kinetic energy term which depends on a mass matrix. 
This is what the `inverse_mass_matrix` is for: It sets the scaling of the proposals in parameter space. A good choice dramatically improves sampling efficiency when parameters have very different scales or are correlated. 
For further information, [Stan's HMC reference](https://mc-stan.org/docs/2_19/reference-manual/hmc-algorithm-parameters.html?) may be interesting. 
This ought to be a quadratic matrix with dimensions equal to the parameter space. Here $p + 1 +1$ for $\beta$, $\mu_\beta$ and $\sigma^2_\beta$, respectively.

In general, `step_size` and `inverse_mass_matrix` should not be set by hand, but by Stan-style window adaption. 

The following code block creates the initial sampler state which does not just store the position, but also caching things required for fast transitions. 
```
# 2) Initialize the state 
initial_position = {
    "beta": jnp.ones(p),
    "log_sigma_sq_beta": 1., 
    "mean_beta": 1.
}
initial_state = nuts.init(initial_position)
state = nuts.init(initial_position)
```
Note the `nuts` object originates in the code block above and contains `step_size` and `inverse_mass_matrix`. 
Note that HMC/NUTS operate on an **unconstrained** Euclidean space! 
If the scale parameter must be positive, the log-scale is (usually) sampled and transformed inside of the posterior log-density function. 

Finally, we run the chain(s): 
```
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
```
As mentioned in [[Jax]], we start with a key for the random functions to consume (which, given the key, return a deterministic value). 
Usually, we use `split` to create a new key, but if we already have a loop index, `fold_in` is preferred as it uses the loop index. 
Finally, we run the chains using [[Jax#Just-in-time compilation]].

To display the chains, specify the burnin-size and run
```
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
```

The [[Posterior Predictive Distribution]] has non-linear boundaries. Not too sure why: Given a fixed $\boldsymbol{\beta}$ vector, the boundary is a hyperplane, i.e. linear. Integrating out may affect this linearity as it is a integral is a mixture of sigmoids (Jensen's inequality)... 

#### Running Example Code ####
```
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
```


