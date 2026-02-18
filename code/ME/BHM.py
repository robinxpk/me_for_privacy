# %%
import jax
import jax.numpy as jnp
import jax.nn as jnn
import blackjax
import matplotlib.pyplot as plt
import pandas as pd

### Draw rng_key for later use
from datetime import date
rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))

# Contains the Bayesian Hierarchical Models (BHMs) class 
class BHM: 
    def __init__(
            self, 
            data: pd.DataFrame, 
            response: str, 
            covariates: list,
            post_log_dens: callable,
            hyperparams: dict,
            initial_positions: dict, 
            inverse_mass_matrix: jnp.ndarray, 
            rng_key: jax.random.PRNGKey,
            num_chains: int = 4,
            inital_step_size: float = 1e-3, 
            init_sampler: callable = blackjax.nuts, 
            warmup_steps = 1_000,
            burnin: int = 1_000,
            n_samples: int = 5_000
        ):

        # The hierarchical model is specified by the posterior log density function in a JAX-compatible way.

        # JAX related attributes
        self.rng_key = rng_key        

        # Design matrix
        self.X = jnp.array(data[covariates].values) 
        self.X = jnp.c_[jnp.ones(self.X.shape[0])[:, None], self.X]
        # Number of covariates (including intercept)
        self.p = self.X.shape[1]
        # Sample size
        self.n = self.X.shape[0]
        # Response variable
        self.y = jnp.array(data[response].values)
        # Latent parameters of the model
        self.params = ""
        # Posterior log density function
        self.logdensity_fn = lambda params: post_log_dens(self.y, self.X, params, **hyperparams)

        # MCMC parameters
        self.initial_positions = initial_positions
        self.num_chains = num_chains
        self.burnin = burnin
        self.n_samples = n_samples
        self.step_size = inital_step_size
        self.inverse_mass_matrix = inverse_mass_matrix
        self.sampler = init_sampler
        self.warmup_steps = warmup_steps
        self.warmup = blackjax.window_adaptation(
            init_sampler,
            self.logdensity_fn
        )

    ### --- Inference Loop
    def _inference_loop(self, rng_key, kernel, initial_state, num_samples):

        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states

    # %%
    def _adapt_and_sample_one_chain(self, rng_key, initial_position):
        
        ## Key handling
        adapt_key, sample_key = jax.random.split(rng_key)
        
        ## For a given initial position, get 
        (state, parameters), info = self.warmup.run(
            adapt_key,
            initial_position,
            num_steps = self.warmup_steps
        )
        kernel = self.sampler(self.logdensity_fn, **parameters)

        chain = self._inference_loop(
            sample_key, 
            kernel.step, 
            state, 
            num_samples = self.n_samples
        )

        return chain
    
    def fit(self): 
        rng_key = jax.random.split(self.rng_key, self.num_chains)
        self.res = jax.vmap(self._adapt_and_sample_one_chain)(rng_key, self.initial_positions)
    
    def viz_chains(self, param_name, _vector_dim = 1, title = ""):
        if len(self.res.position[param_name].shape) != 2: _vector_dim = self.res.position[param_name].shape[2]
        fig, ax = plt.subplots(1, _vector_dim, figsize=(12, 2))
        # To esnure the matplotlib axes are always iterable, even if only one parameter is plotted
        if _vector_dim == 1: ax = [ax]

        for i, axi in enumerate(ax):
            for chain in range(self.num_chains):
                if _vector_dim == 1:
                    axi.plot(self.res.position[param_name][chain], alpha = 0.5)
                else:
                    axi.plot(self.res.position[param_name][chain][:, i], alpha = 0.5)
            axi.axvline(x = self.burnin, color = "red")
            # Latex formatting 
            # axi.set_title(f"${param_name}{i}$")
            axi.set_title(f"{param_name}{i}")
        plt.show()
    
    def mean_estimates(self, param_name): 
        # TODO: This currently also includes the burnin. Should be removed using self.burnin attribute.
        return jnp.array([
            self.res.position[param_name][chain].mean(axis = 0)  for chain in range(self.num_chains)
        ]).mean(axis = 0)