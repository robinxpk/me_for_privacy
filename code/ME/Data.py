import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm

class Data:
    def __init__(
        self, raw_data, prob,
        seed = 1234, error_factors = np.array([1])
    ):
        self.seed = seed
        self.prob = prob

        self.raw_data = raw_data
        self.shape = raw_data.shape
        self.mask_bool = self._create_mask_bool()
        self.error_vars = self._draw_error_vars(error_factors = error_factors)

        self.masked_data = None
        # Fill masked data:
        self._mask_raw_data()

    def _create_mask_bool(self):
        n, p = self.shape
        rng = np.random.default_rng(self.seed)
        bools_matrix = rng.random((n, p)) < self.prob
        return(pd.DataFrame(bools_matrix, columns = self.raw_data.columns))
    
    def _draw_error_vars(
            self,
            error_factors = np.array([5])
        ):
        n, p = self.shape
        rng = np.random.default_rng(self.seed)
        error_vars = rng.choice(error_factors, size = (n, p), replace = True)

        return pd.DataFrame(error_vars, columns = self.raw_data.columns)
    
    def _mask_raw_data(self):
        self.masked_data = self.raw_data.apply(self._mask_raw_column, axis = 0) # axis = 0 --> column wise

    def _mask_raw_column(self, col):
        to_mask = self.mask_bool[col.name]

        if col.dtype == np.dtype("float64"):
            # Numerical variable
            # ePIT auf allen Variable bestimmen
            

            #  und Variable in Normalverteilung transformieren

            #         Normalen Fehlerterm aufaddieren: 
            #         Fehlervarianz randomly ziehen
            #         Grund für random draw: Kein gesondertes Behandeln der Outlier
            #             sonder es kann durch eine sehr hohe Fehlervarianz auch zu Outliern kommen. So erhalten Outlier nicht nur möglicherweise einen Fehlerterm, sondern bestehende Outlier werden durch mögliche neue Outlier "gemasked". 
            #             Natürlich kann es auch dazu kommen, dass Outlier noch weiter in die Outlier-Richtung gezogen werden. Hier machen wir uns aber erstmal keine Sorgen, weil das durch gutes Masking (hoffentlich) weniger relevant ist.
            ecdf = ECDF(col.values) # Inits eCDF based on all input values
            pobs = ecdf(col.values[to_mask]) * len(col) / (len(col) + 1) # Determines pseudo observations for observed values that are supposed to have error added on; Note: Use len(.) + 1 to not have values == 1. ecdf does not implicitly do this...
            pobs = np.clip(pobs, 1e-12, 1 - 1e-12) # Avoid exactly 0 or 1
            std_normal_obs = norm.ppf(pobs)
            error_obs = std_normal_obs + np.random.normal(loc = 0, scale = self.error_vars[col.name][to_mask])
            pobs_error = norm.cdf(error_obs)

            # Self-implement backtransition from pseudo-values bc python seems to not have an implemented solution
            sorted_vals = np.sort(col.values)

            k = np.floor(pobs_error * (len(col) + 1.0)).astype(int)
            k = np.clip(k, 1, len(col))

            obs_with_error = sorted_vals[k - 1]  # 0-based indexing

            # Replace values with values + error
            # col[to_mask] = obs_with_error 
            col_error = col.copy() 
            col_error[to_mask] = obs_with_error
        elif col.dtype == np.dtype("str"):
            # Categorical variable
            pass
        return col_error
    
    def _apply_numerical_error(self, pobs, var):
        pass

    


