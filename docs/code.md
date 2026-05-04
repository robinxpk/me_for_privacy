## Implementation Notes ##
### Evaluation of Bias due to ME ###
Code: 
- a) Iterate through each error and </br> 
    b) (possibly) different error variances and </br> 
    c) the $B$ data sets. 
    - Run in parallel for sure, but worried about clash and output and all. 

#### Important aspects and some reasons for why its implemented as it is ####
- **DO NOT** use`fork` to parallelize, but `spawn`. 
Else, the OS might struggle (deadlock). 
    - TODO: Not sure what a deadlock is. lol.
- Use CPU parallel only because I think the current server does not have any GPU. Also, not sure if GPU would even give that much of a speedup, because currently, the matrices are manageable. Think GPU might be more relevant if we move to more intense matrix calculation. 
- Nested parallelism seems odd for a multitude of reasons: 
    - Writing into file? (not actually a reason against nested parallel, but something to worry about when implementing)
    - `JAX` is multithreaded and (somehow) already uses multiple cores. Weird, threading is not parallelizing, right? </br> 
    Whatever. Due to this, set the threads to $1$ **BEFORE** importing **ANY** `JAX` related package. </br> 
    This is done using `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`. 
Or use the following in the python script before importing jax. Within script is a temp solution IMO, because this should be dealt with in shell.
```
# %%
import os
# BEFORE important jax, set XLA flags to disable threading such that data sets can run in parallel (hopefully) without issues
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" 
```
- TODO: Read up on what threading in `JAX` means.
    - See ["As a result, each step is as long as the slowest chain."](https://blackjax.readthedocs.io/en/latest/examples/howto_sample_multiple_chains.html), i.e. for threads and NUTS: Chains are not run as independent workers(?). Instead, the threads basically wait for the slowest chain. So overall, the script is as slow as the slowest chain. 
- Accelerated Linear Algebra ([XLA](https://docs.jax.dev/en/latest/xla_flags.html)) is "the powerhouse" of `JAX`. 
    - Set XLA flags are environment variables that influence how JAX is executed. 
- Remove the whole burnin phase because this is already done in the STAN warmup step


# TODO #
- Quantify effect of distributional-misspecification (e.g. of the distribution of the real data). 
