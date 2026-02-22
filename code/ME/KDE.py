import jax.numpy as jnp

# This class holds the KDE model, which is used to estimate the empirical density of the true covariates.
# Whenever using the empirical density of the data in JAX, it is required to fulfil the jax primitives. 
# Thus, use this class as duck typing when needed or to abstract logic for, e.g. fitting a KDE oneself USING JAX. (?)

# NOTE: The (currently used) JAX KDE "gaussian_kde" relies on the TRANSPOSED X matrix. wtf. 
# In purpose of ducktyping, use the same convention for the input X in the KDE model, i.e. X is transposed compared to the usual design matrix.
    #i. e.: 
    # 1) Use the TRANSPOSED X as input for the KDE model, such that it is compatible with the JAX KDE "gaussian_kde".
    # 2) Use the TRANSPOSED X as input for the evaluate method



class KDE_Model():
    # This class is only for KDEs to interhit from. It holds the most important method: evaluate, which is used inthe log-posterior density and evaluates the KDE on a given input X. 
    def __init__(self, data_transposed): 
        self.data = data_transposed
    
    def _logpdf(self, X_transposed):
        raise NotImplementedError("Subclasses must implement _evaluate(X_transposed).")

    def logpdf(self, X_transposed): 
        # Evaluate the KDE on the input (vector / matrix) X. 
        # Returns a vector of density values for each row of X. 
        return self._logpdf(X_transposed)

class KDE_Dummy_Model(KDE_Model):
    # This is the dummy version of the KDE model, which returns constant 0s for the empirical log density, such that it drops from the log density sum.
    # i.e. it allows to easily switch between using an empirical density or not, without having to change the structure of the log density function.

    def __init__(self): 
        # No data is needed for the dummy model, since it returns constant 0s.
        pass

    def _logpdf(self, X_transposed): 
        return jnp.zeros(X_transposed.shape[1])