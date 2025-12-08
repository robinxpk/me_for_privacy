import numpy as np
import umap
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from ME.Cluster import Cluster
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
        
class Data:
    def __init__(
        self, raw_data, prob,
        seed = 1234, error_factors = np.array([1]),
        error_type = "ePIT", cluster_based = False
    ):
        self.seed = seed
        self.prob = prob
        self.error_type = error_type

        self.raw_data = raw_data
        self.n = len(self.raw_data.index)
        self.shape = raw_data.shape
        self.mask_bool = self._create_mask_bool()
        self.error_vars = self._draw_error_vars(error_factors = error_factors)

        self.masked_data = None
        self.cluster_based = cluster_based
        self.prior_cluster = self._assign_cluster(data = self.raw_data, type = "k-means")
        # Fill masked data:
        self._mask_raw_data()
        self.post_cluster = self._assign_cluster(data = self.masked_data, type = "k-means")

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
        if self.error_type == "ePIT": 
            # Error: var -> pobs -> std_norm + norm_error -> pobs -> inv-ePIT
            self.masked_data = self.raw_data.apply(self._apply_ePIT_error, axis = 0) # axis = 0 --> column wise
        elif self.error_type == "lognormal": 
            # var * exp(normal) [multiplicative] bzw. log(var) + normal_error; ! Only applicable to numericals
            self.masked_data = self.raw_data.apply(self._apply_lognormal_error, axis = 0)
        elif self.error_type == "normal": 
            # var + normal [additive]; ! Only applicable to numericals
            self.masked_data = self.raw_data.apply(self._apply_normal_error, axis = 0)
        elif self.error_type == "berkson": 
            self.cluster_based = True
            self.masked_data = self.raw_data.apply(self._apply_berkson_error, axis = 0)
    
    def _apply_berkson_error(self, col): 
        if col.dtype.name == "category":
            return(col)
        to_mask = self.mask_bool[col.name]
        col_error = col.copy()
        
        cluster_idx = list(self.prior_cluster.names).index(col.name)
        col_error[to_mask] = self.prior_cluster.fit.cluster_centers_[self.prior_cluster.fit.labels_[to_mask], cluster_idx]

        return col_error
    
    def _assign_cluster(self, data,  type = "k-means", n_neighbors = 100):
        """
        For cluster based error, fit the cluster on the original data. This will then be used to apply the error-structure.
        """
        if type != "k-means": 
            raise ValueError(f"The prior cluster type is not 'k-means-constrained'[was ", type,"instead]")
        working_df = data.select_dtypes(include="number")
         
        # reducer = umap.UMAP(n_neighbors = n_neighbors)
        # # Standardize data to not have the result depend on scale of the variable
        # scaled_data = StandardScaler().fit_transform(working_df)
        # prior_embedding = reducer.fit_transform(scaled_data)
        
        # plt.figure()
        # plt.scatter(prior_embedding[:, 0], prior_embedding[:, 1], alpha = 0.1)
        # plt.title('UMAP projection of the raw data', fontsize=18)
        # plt.savefig(f"BerksonError_UmapOnRawData_neighbors{n_neighbors}.png")

        k_means = sklearn.cluster.KMeans(n_clusters = (self.n // n_neighbors) + 1)
        k_means.fit(np.array(working_df))
        # k_means.names = working_df.columns
        return(Cluster(fit = k_means, data = working_df))

        # plt.figure()
        # plt.scatter(prior_embedding[:, 0], prior_embedding[:, 1], alpha = 0.3, c = k_means.labels_)
        # plt.title('UMAP projection of the raw data', fontsize=18)
        # plt.savefig(f"BerksonError_LabelledUmapOnRawData_neighbors{n_neighbors}.png")



    def _apply_lognormal_error(self, col):    
        if col.dtype.name == "category":
            return(col)
        to_mask = self.mask_bool[col.name]
        col_error = col.copy()

        n_errors = sum(to_mask)
        mu = np.zeros((n_errors,))
        var = np.array(self.error_vars[col.name][to_mask])
        # NOTE: Do not draw from np.random.multivariate_normal bc it creates the full covariance matrix which goes crazy in memory
            # Instead, univariate normal allows vector of variances --> draws with different variances
        norm_error = np.random.normal(loc = mu, scale = var, size = n_errors)
        col_error[to_mask] = col[to_mask] * np.exp(norm_error)

        return col_error

    def _apply_normal_error(self, col):
        if col.dtype.name == "category":
            return(col)
        rng = np.random.default_rng(self.seed)
        to_mask = self.mask_bool[col.name]
        col_error = col.copy()

        n_errors = sum(to_mask)
        mu = np.zeros((n_errors,))
        var = np.array(self.error_vars[col.name][to_mask])

        norm_error = np.random.normal(loc = mu, scale = var, size = n_errors)
        col_error[to_mask] = col[to_mask] + norm_error

        # If col does not contain negative values, assume non-negativity and replace negative values with non-negative ones
            # Randomly draw from Lowest 5% in column
        if col.min() >= 0: 
            negative_cases = col_error[col_error < 0]
            q05 = np.quantile(col, 0.05)       # 5% empirical quantile
            smallest_5 = col[col <= q05] 
            col_error[col_error < 0] = rng.choice(smallest_5, size = len(negative_cases), replace=True) 

        return col_error

    def _apply_ePIT_error(self, col):
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
        elif col.dtype.name == "category":
            # Categorical variable
            base_values = col.unique()
            random_ints = np.random.random_integers(low = 0, high = len(base_values) - 1, size = len(col[to_mask]))

            col_error = col.copy() 
            col_error[to_mask] = [base_values[random_draw] for random_draw in random_ints]
        return col_error
    
    def drop_column(self, colname):
        self.raw_data = self.raw_data.drop(colname, axis = 1)
        self.masked_data = self.masked_data.drop(colname, axis = 1)
    
    def filter_cluster_raw(self, cluster): 
        return self.raw_data[self.raw_filter == cluster]

    def viz_error_effect(self, varname):
        if self.raw_data[varname].dtype.name == "category": 
            fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            self.raw_data.plot(kind = "bar", ax = ax[0])
            ax[0].set_title("Barplot raw data")
            self.masked_data.plot(kind = "bar", ax = ax[1])
            ax[1].set_title(f"Barplot WITH error [{self.error_type}]")
            ax.legend()
            ax.set_xlabel(varname)
            plt.tight_layout()
            plt.show()
        elif self.raw_data[varname].dtype == np.dtype("float64"):
            ax = self.raw_data[varname].plot.kde(label="Raw Data")          
            self.masked_data[varname].plot.kde(ax=ax, label="Data w/Error", style = "--")  
            ax.set_title(f'KDE w/ and w/o error [{self.error_type}]')
            ax.set_xlabel(varname)
            ax.legend()
            plt.show() 

    def ePIT_viz(self, varname): 
        if self.error_type != "ePIT": 
            return

        # 1) Data and empirical CDF / PIT
        s = self.raw_data[varname].dropna()
        x = s.values
        n = len(x)

        # sorted x and empirical CDF (for plotting)
        x_sorted = np.sort(x)
        F_emp = np.arange(1, n + 1) / (n + 1)   # ECDF values

        # PIT values for each observation (Uniform(0,1) approx.)
        u = s.rank(method="average").values / (n + 1)

        # transform U ~ approx Uniform(0,1) to Z ~ approx N(0,1)
        z = norm.ppf(u)

        fig, axes = plt.subplots(2, 2, figsize=(10, 6))

        # --- top left: histogram of F̂_X(X) (≈ Uniform(0,1)) ---
        axes[0, 0].hist(u, bins=20, density=True, orientation="horizontal" )
        axes[0, 0].set_title(r"hist of $\hat F_X(X)$")
        axes[0, 0].set_xlabel("u")
        axes[0, 0].set_ylabel("density")

        # --- top right: empirical CDF ---
        axes[0, 1].plot(x_sorted, F_emp)
        axes[0, 1].set_title(r"Empirical CDF $\hat F_X(x)$")
        axes[0, 1].set_xlabel("x")
        axes[0, 1].set_ylabel("F")

        # --- bottom right: empirical density via KDE ---
        s.plot.kde(ax=axes[1, 1])
        axes[1, 1].set_title(r"Empirical density $\hat f_X(x)$")
        axes[1, 1].set_xlabel("x")
        axes[1, 1].set_ylabel("density")

        # --- bottom left: empty (to mimic the L-shape layout) ---
        axes[1, 0].axis("off")

        plt.tight_layout()
        plt.show()

        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 6))

        # --- top left: histogram of U (same PIT as before) ---
        axes2[0, 0].hist(u, bins=20, density=True, orientation="horizontal" )
        axes2[0, 0].set_title(r"hist of $U$ (PIT)")
        axes2[0, 0].set_xlabel("u")
        axes2[0, 0].set_ylabel("density")

        # --- top right: standard normal CDF Φ(z) ---
        z_grid = np.linspace(-4, 4, 400)
        axes2[0, 1].plot(z_grid, norm.cdf(z_grid))
        axes2[0, 1].set_title(r"Standard normal CDF $\Phi(z)$")
        axes2[0, 1].set_xlabel("z")
        axes2[0, 1].set_ylabel("Φ(z)")

        # --- bottom right: standard normal density φ(z), with histogram of Z ---
        axes2[1, 1].hist(z, bins=20, density=True, alpha=0.5, label="empirical Z")
        axes2[1, 1].plot(z_grid, norm.pdf(z_grid), label="N(0,1) pdf")
        axes2[1, 1].set_title(r"Standard normal density $\varphi(z)$")
        axes2[1, 1].set_xlabel("z")
        axes2[1, 1].set_ylabel("density")
        axes2[1, 1].legend()

        # --- bottom left: empty again ---
        axes2[1, 0].axis("off")

        plt.tight_layout()
        plt.show()
    
    def compare_cluster(): 
        pass
