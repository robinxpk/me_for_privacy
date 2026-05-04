import os 
import jax
import jax.numpy as jnp
import blackjax
import pandas as pd

from ME.BHM import BHM
from ME.Data import Data
from ME.KDE import KDE_Dummy_Model
from jax.scipy.stats import gaussian_kde

from ME.functions import post_log_dens, post_log_dens_gaussian_additive, post_log_dens_lognormal_multiplicative, post_log_dens_epit

#!! ----------------------- Auxiliary Functions --------------------------------------- !!#
### #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# ###

def single_iteration(
    # TODO: Add comment for each input 
        b, 
        error_name, 
        error_variance, 
        empirical_kde_mdl,
        raw_data, 
        num_chains, 
        covariates,
        response,
        error_subset,
        p,
        n_samples,
        n_warmup_steps, 
        corrected_post_log_dens, 

        rng_key
    ): 
    # Define these functions here in somewhat "hardcoded" way because of how windows spawns it workers. 
    #-> Section implementing the super random sigmoid function I decided to use to proxi the empirical cdf to have continuous and and nice proxi und alles
    # Estimates: 
    # beta_0 = -30 
    # beta_1 = 4
    # --> See R (load_data.R script; at the bottom of the file) Nice mess of files now. 
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
    # KDE Model 
    dummy_empirical_kde_mdl = KDE_Dummy_Model()

    # A single interation consists of: 
    # 1) Create a new data frame
    # 2) Fit a naive BHM
    # 3) Fit a BHM accounting for ME 
    # 4) Save results in csv file

    # The b-th data set is uses default seed + b. Rest is either given or constant. 
    voe_data = Data(
        name = f"{error_name}_{list(error_variance.values())[0]}", 
        raw_data = raw_data, 
        seed = 1234 + b, # Adding b on same base seed to ensure different data sets across iterations, but same data sets across models (naive and corrected) within the same iteration.
        error_vars = error_variance, 
        error_type = error_name,
        cols_excluded_from_error = ["LBXT4", "RIDAGEYR", "bmi"],
        e_sigmoid = e_sigmoid, 
        e_inv_sigmoid = e_inv_sigmoid
    )
    
    # Because my current implementation of ePIT is terrible, this if case must be present. yikes. NEED TO fix this!!
    if error_name == "ePIT": 
        ### The epit model is super cooked... 
        # Epit Adjustments: 
        # --- 1) Express data fully in terms of tilde(z) instead of tilde(x)
        voe_data.masked_data.loc[:, "DR1TKCAL"] = jax.scipy.stats.norm.ppf(e_sigmoid(voe_data.masked_data.loc[:, "DR1TKCAL"].values)) 
        # --- 2) Express KDE in terms of (true) z
        # Because I draw z, the KDE must be expressed in terms of Z, too. 
        # Basically, the design matrix is X, but I evaluate partially X and z where touched by error
        data_raw_with_z = voe_data.raw_data
        data_raw_with_z.loc[:, "DR1TKCAL"] = jax.scipy.stats.norm.ppf(e_sigmoid(voe_data.raw_data.loc[:, "DR1TKCAL"].values)) 
        empirical_kde_mdl = gaussian_kde(data_raw_with_z.loc[:, covariates].values.T, bw_method = "scott")
        # --- 3) Express observed values in terms of tilde(z) instead of tilde(x) for initial values
        # ! When sub-selecting a columns, use this notation. Else, the code breaks.
        init_vals = {
            # Beta estimates
            "beta": jnp.zeros((num_chains, p)),
            # Use empirical variance of response as initial value
            "log_sigma": jnp.repeat(jnp.log(jnp.var(jnp.asarray(data[response].values))), num_chains, axis = 0), 
            # For true observed values, start off with the ERROR-CONTAMINATED values only
            "Z_true": jnp.tile(voe_data.masked_data[error_subset].values, (num_chains, 1, 1)) 
        }
        ###
    else:
        init_vals = {
            # Beta estimates
            "beta": jnp.zeros((num_chains, p)),
            # Use empirical variance of response as initial value
            "log_sigma": jnp.repeat(jnp.log(jnp.var(jnp.asarray(raw_data[response].values))), num_chains, axis = 0), 
            # For true observed values, start off with the ERROR-CONTAMINATED values only
            "X_true": jnp.tile(voe_data.masked_data[error_subset].values, (num_chains, 1, 1)) 
        }
    
    # Fit native Bayesian Hierarchical Model
    naive = BHM(
        data = voe_data.masked_data,
        response = response,
        error_cols = [],
        covariates = covariates,
        post_log_dens = post_log_dens,
        hyperparams = {
            "b": 100, # TODO: Depends on scale, of course. e.g. age reaches up to 80, calories reach up to 4k, bmi reaches up to 40; this is the SD of the posterior
            "c": 1,
            "d": 1
        },
        initial_positions = {
            "beta": jnp.zeros((num_chains, p)),
            "log_sigma": jnp.ones((num_chains, ))
        }, 
        empirical_kde_mdl = dummy_empirical_kde_mdl,
        error_cov_matrix = jnp.diag(jnp.array([])), 
        inverse_mass_matrix = jnp.eye(3),
        rng_key = rng_key,
        num_chains = num_chains,
        inital_step_size = 1e-3,
        # Burnin and warmup are default values with 1_000
        n_samples = n_samples
    )

    corrected = BHM(
        data = voe_data.masked_data,
        response = response,
        covariates = covariates,
        # JAX does not allow me to pass error_cols as string to index the column in the design matrix touched by error. 
        # Because internally, the design matrix is a jnp array and not a dataframe, I need to pass the column index instead of the column name.
        error_cols = error_subset, 
        post_log_dens = corrected_post_log_dens[error_name],
        hyperparams = {
            "b": 100, # TODO: Depends on scale, of course. e.g. age reaches up to 80, calories reach up to 4k, bmi reaches up to 40; this is the SD of the posterior
            "c": 1,
            "d": 1
        },
        empirical_kde_mdl = empirical_kde_mdl,
        error_cov_matrix = jnp.diag(jnp.array(list(error_variance.values()))) ,
        initial_positions = init_vals, 
        inverse_mass_matrix = jnp.eye(3),
        rng_key = rng_key,
        num_chains = num_chains,
        inital_step_size = 1e-3, 
        warmup_steps = n_warmup_steps, 
        n_samples = n_samples, 
        e_sigmoid = e_sigmoid, 
        e_inv_sigmoid = e_inv_sigmoid
    )

    naive.fit()
    corrected.fit()

    ## Save results
    # Get R hat
    naive_rhat_beta = blackjax.diagnostics.potential_scale_reduction(naive.res.position["beta"])
    naive_rhat_log_sigma = blackjax.diagnostics.potential_scale_reduction(naive.res.position["log_sigma"])
    corrected_rhat_beta = blackjax.diagnostics.potential_scale_reduction(corrected.res.position["beta"])
    corrected_rhat_log_sigma = blackjax.diagnostics.potential_scale_reduction(corrected.res.position["log_sigma"])

    # Get confidence intervals
    naive_beta_ci = naive.confidence_interval("beta")
    naive_log_sigma_ci = naive.confidence_interval("log_sigma")
    corrected_beta_ci = corrected.confidence_interval("beta")
    corrected_log_sigma_ci = corrected.confidence_interval("log_sigma")

    # Hardcoded results. yey.
    res = pd.DataFrame(
        data = {
            # Metadata
            "error":            [error_name                                 ],
            "error_variance":   [float(list(error_variance.values())[0])    ],
            "b":                [int(b)                                     ],

            # Naive Model Estimates
            "naive_beta0":      [float(naive.mean_estimates(param_name = "beta")[0])    ], 
            "naive_beta1":      [float(naive.mean_estimates(param_name = "beta")[1])    ],
            "naive_beta2":      [float(naive.mean_estimates(param_name = "beta")[2])    ],
            "naive_beta3":      [float(naive.mean_estimates(param_name = "beta")[3])    ], 
            "naive_log_sigma":  [float(naive.mean_estimates(param_name = "log_sigma"))  ], 

            "naive_rhat_beta0":     [float(naive_rhat_beta[0])  ],
            "naive_rhat_beta1":     [float(naive_rhat_beta[1])  ],
            "naive_rhat_beta2":     [float(naive_rhat_beta[2])  ],
            "naive_rhat_beta3":     [float(naive_rhat_beta[3])  ],
            "naive_rhat_log_sigma": [float(naive_rhat_log_sigma)],

            # Naive Model CIs
            "naive_beta0_ci_lower": [float(naive_beta_ci["lower"][0])],
            "naive_beta0_ci_upper": [float(naive_beta_ci["upper"][0])],
            "naive_beta1_ci_lower": [float(naive_beta_ci["lower"][1])],
            "naive_beta1_ci_upper": [float(naive_beta_ci["upper"][1])],
            "naive_beta2_ci_lower": [float(naive_beta_ci["lower"][2])],
            "naive_beta2_ci_upper": [float(naive_beta_ci["upper"][2])],
            "naive_beta3_ci_lower": [float(naive_beta_ci["lower"][3])],
            "naive_beta3_ci_upper": [float(naive_beta_ci["upper"][3])],
            "naive_log_sigma_ci_lower": [float(naive_log_sigma_ci["lower"])],
            "naive_log_sigma_ci_upper": [float(naive_log_sigma_ci["upper"])],

            # Corrected Model Estimates
            "corrected_beta0":      [float(corrected.mean_estimates(param_name = "beta")[0])    ], 
            "corrected_beta1":      [float(corrected.mean_estimates(param_name = "beta")[1])    ],
            "corrected_beta2":      [float(corrected.mean_estimates(param_name = "beta")[2])    ],
            "corrected_beta3":      [float(corrected.mean_estimates(param_name = "beta")[3])    ], 
            "corrected_log_sigma":  [float(corrected.mean_estimates(param_name = "log_sigma"))  ], 

            "corrected_rhat_beta0":     [float(corrected_rhat_beta[0])  ],
            "corrected_rhat_beta1":     [float(corrected_rhat_beta[1])  ],
            "corrected_rhat_beta2":     [float(corrected_rhat_beta[2])  ],
            "corrected_rhat_beta3":     [float(corrected_rhat_beta[3])  ],
            "corrected_rhat_log_sigma": [float(corrected_rhat_log_sigma)] ,

            # Corrected Model CIs
            "corrected_beta0_ci_lower": [float(corrected_beta_ci["lower"][0])],
            "corrected_beta0_ci_upper": [float(corrected_beta_ci["upper"][0])],
            "corrected_beta1_ci_lower": [float(corrected_beta_ci["lower"][1])],
            "corrected_beta1_ci_upper": [float(corrected_beta_ci["upper"][1])],
            "corrected_beta2_ci_lower": [float(corrected_beta_ci["lower"][2])],
            "corrected_beta2_ci_upper": [float(corrected_beta_ci["upper"][2])],
            "corrected_beta3_ci_lower": [float(corrected_beta_ci["lower"][3])],
            "corrected_beta3_ci_upper": [float(corrected_beta_ci["upper"][3])],
            "corrected_log_sigma_ci_lower": [float(corrected_log_sigma_ci["lower"])],
            "corrected_log_sigma_ci_upper": [float(corrected_log_sigma_ci["upper"])]
        }
    )
    filename = f"out_{error_name}_{list(error_variance.values())[0]}_{b}.csv"
    res.to_csv(filename, sep = ";", index = False)

def build_args(
        B: int, 
        **kwargs
    ): 
    # The starmap function allows to pass tuples of function inputs in the multiprocessing steps, e.g.: 
    # pool.starmap(g, [(1, 10), (2, 20), (3, 30)])
    # g(1, 10), g(2, 20), g(3, 30)
    # This function builds the required tuples 
    # Output shape: 
    # [..., (b_i, error_name, error_variance), ...] --> order in which the function single_iteration expects it

    return [(tuple([b] + list(kwargs.values()))) for b in range(B)]

def fit_data_in_parallel(
        error_name, 
        error_variance, 
        B, 
        empirical_kde_mdl, 
        raw_data, 
        num_chains, 
        covariates,
        response,
        error_subset,
        p,
        n_samples,
        n_warmup_steps, 
        corrected_post_log_dens, 

        rng_key, 

        ctx
    ): 
    # TODO: Write this function such that it applies the loop in parallel and save results in an output file
    # Create a total of B data sets and fit naive as well as corrected model
    workers = min(B, max(1, os.cpu_count() - 4) or 1)
    args = build_args(
            B = B, 
            # !! Order of the following variable must be the same as single_iteration expects them.
            error_name = error_name, 
            error_variance = error_variance, 
            empirical_kde_mdl = empirical_kde_mdl, 
            raw_data = raw_data, 
            num_chains = num_chains, 
            covariates = covariates,
            response = response,
            error_subset = error_subset,
            p = p, 
            n_samples = n_samples,
            n_warmup_steps = n_warmup_steps, 
            corrected_post_log_dens = corrected_post_log_dens, 
            rng_key = rng_key
        )

    with ctx.Pool(processes = workers) as pool: 
        args = args
        # Automatically writes output
        pool.starmap(single_iteration, args)
    