import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import invgauss

class Data:
    def __init__(
        self, raw_data, prob,
        seed = 1234
    ):
        self.seed = seed
        self.prob = prob

        self.raw_data = raw_data
        self.shape = raw_data.shape
        self.mask_bool = self._create_mask_bool()
        self.error_vars = self._draw_error_vars()

        self.mask_data = self._mask_raw_data()

    def _create_mask_bool(self):
        n, p = self.shape
        rng = np.random.default_rng(self.seed)
        bools_matrix = rng.random((n, p)) < self.prob
        return(pd.DataFrame(bools_matrix, columns = self.raw_data.columns))
    
    def _draw_error_vars(
            self,
            error_factors = np.array([1, 1.5, 2])
        ):
        n, p = self.shape
        rng = np.random.default_rng(self.seed)

        return rng.choice(error_factors, size = (n, p), replace = True)
    
    def _mask_raw_data(self):
        return self.raw_data.apply(self._mask_raw_column, axis = 0) # axis = 0 --> column wise

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
            ecdf = ECDF(col.values) # Inits eCDF
            pobs = ecdf(col.values) # Determines pseudo observations
            self._apply_numerical_error(pobs.loc[to_mask], var = col.var())

            
            pass
        elif col.dtype == np.dtype("str"):
            # Categorical variable
            pass
        return "print"
    
    def _apply_numerical_error(self, pobs, var):
        invgauss(pobs, mu = 0, scale = 1)
        pass

    


