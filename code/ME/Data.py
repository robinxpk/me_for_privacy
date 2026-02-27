import numpy as np
import umap
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from ME.Cluster import Cluster
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler


class MeasurementErrorModel:
    """Base class for measurement error models."""

    error_type = "base"

    def apply_to_column(self, data_obj, col):
        raise NotImplementedError("Subclasses must implement apply_to_column(data_obj, col).")


class EPITErrorModel(MeasurementErrorModel):
    error_type = "ePIT"

    def apply_to_column(self, data_obj, col):
        to_mask = data_obj.mask_bool[col.name]
        col_error = col.copy()

        if col.dtype == np.dtype("float64"):
            ecdf = ECDF(col.values)
            pobs = ecdf(col.values[to_mask]) * len(col) / (len(col) + 1)
            pobs = np.clip(pobs, 1e-12, 1 - 1e-12)
            std_normal_obs = norm.ppf(pobs)
            error_obs = std_normal_obs + np.random.normal(
                loc=0,
                scale=data_obj.error_vars[col.name][to_mask],
            )
            pobs_error = norm.cdf(error_obs)

            sorted_vals = np.sort(col.values)
            k = np.floor(pobs_error * (len(col) + 1.0)).astype(int)
            k = np.clip(k, 1, len(col))

            obs_with_error = sorted_vals[k - 1]
            col_error[to_mask] = obs_with_error

        elif col.dtype.name == "category":
            base_values = col.unique()
            random_ints = np.random.randint(
                low=0,
                high=len(base_values),
                size=len(col[to_mask]),
            )
            col_error[to_mask] = [base_values[random_draw] for random_draw in random_ints]

        return col_error


class LognormalErrorModel(MeasurementErrorModel):
    error_type = "lognormal"

    def apply_to_column(self, data_obj, col):
        if col.dtype.name == "category":
            return col

        np.random.seed(data_obj.seed)

        to_mask = data_obj.mask_bool[col.name]
        col_error = col.copy()

        n_errors = sum(to_mask)
        var = np.array(data_obj.error_vars[col.name][to_mask])
        mu = -var / 2
        norm_error = np.random.normal(loc=mu, scale=var, size=n_errors)
        error = col[to_mask] * np.exp(norm_error)
        if col.dtype == "int64":
            error = np.floor(error)
        col_error[to_mask] = error

        return col_error


class NormalErrorModel(MeasurementErrorModel):
    error_type = "normal"

    def apply_to_column(self, data_obj, col):
        if col.dtype.name == "category":
            return col

        rng = np.random.default_rng(data_obj.seed)
        to_mask = data_obj.mask_bool[col.name]
        col_error = col.copy()

        n_errors = sum(to_mask)
        mu = np.zeros((n_errors,))
        var = np.array(data_obj.error_vars[col.name][to_mask])

        np.random.seed(data_obj.seed)
        norm_error = np.random.normal(loc=mu, scale=var, size=n_errors)

        error = col[to_mask] + norm_error
        if col.dtype == "int64":
            error = np.floor(error)
        col_error[to_mask] = error

        if col.min() >= 0:
            negative_cases = col_error[col_error < 0]
            q05 = np.quantile(col, 0.05)
            smallest_5 = col[col <= q05]
            col_error[col_error < 0] = rng.choice(smallest_5, size=len(negative_cases), replace=True)

        return col_error


class BerksonErrorModel(MeasurementErrorModel):
    error_type = "berkson"

    def apply_to_column(self, data_obj, col):
        if col.dtype.name == "category":
            return col

        to_mask = data_obj.mask_bool[col.name]
        col_error = col.copy()

        cluster_idx = list(data_obj.prior_cluster.names).index(col.name)
        if col.dtype == "int64":
            data_obj.prior_cluster.fit.cluster_centers_[
                data_obj.prior_cluster.fit.labels_[to_mask], cluster_idx
            ] = data_obj.prior_cluster.fit.cluster_centers_[
                data_obj.prior_cluster.fit.labels_[to_mask], cluster_idx
            ].astype("int64")
        col_error[to_mask] = data_obj.prior_cluster.fit.cluster_centers_[
            data_obj.prior_cluster.fit.labels_[to_mask], cluster_idx
        ]

        return col_error


class Data:
    ERROR_MODELS = {
        EPITErrorModel.error_type: EPITErrorModel,
        LognormalErrorModel.error_type: LognormalErrorModel,
        NormalErrorModel.error_type: NormalErrorModel,
        BerksonErrorModel.error_type: BerksonErrorModel,
    }

    def __init__(
        self,
        name,
        raw_data,
        prob,
        seed=1234,
        error_factors=np.array([1]),
        error_type="ePIT",
        cluster_based=False,
        cols_excluded_from_error=[],
    ):
        self.name = name
        self.seed = seed
        self.prob = prob
        self.error_type = error_type
        self.excluded_cols = cols_excluded_from_error

        self.raw_data = raw_data
        self.n = len(self.raw_data.index)
        self.shape = raw_data.shape
        self.mask_bool = self._create_mask_bool()
        self.error_vars = self._draw_error_vars(error_factors=error_factors)
        self.error_model = self._build_error_model()

        self.masked_data = None
        self.cluster_based = cluster_based
        self.prior_cluster = self._assign_cluster(data=self.raw_data, type="k-means")
        self._mask_raw_data()
        self.post_cluster = self._assign_cluster(data=self.masked_data, type="k-means")
        self.error_evaluation = self.evaluate_errors()

        self.true_var = self.raw_data.select_dtypes(include="number").var()
        self.masked_var = self.masked_data.select_dtypes(include="number").var()

    def evaluate_errors(self):
        sqd_diffs = (
            self.raw_data.select_dtypes(include="number")
            - self.masked_data.select_dtypes(include="number")
        ) ** 2
        mse = sqd_diffs.mean()
        nmse = mse / self.raw_data.select_dtypes(include="number").var()
        return nmse

    def _create_mask_bool(self):
        n, p = self.shape
        rng = np.random.default_rng(self.seed)
        bools_matrix = rng.random((n, p)) < self.prob
        bools_df = pd.DataFrame(bools_matrix, columns=self.raw_data.columns)
        bools_df.loc[:, self.excluded_cols] = False
        return bools_df

    def _draw_error_vars(self, error_factors=np.array([5])):
        n, p = self.shape
        rng = np.random.default_rng(self.seed)
        error_vars = rng.choice(error_factors, size=(n, p), replace=True)

        return pd.DataFrame(error_vars, columns=self.raw_data.columns)

    def _build_error_model(self):
        error_model_cls = self.ERROR_MODELS.get(self.error_type)
        if error_model_cls is None:
            available = ", ".join(sorted(self.ERROR_MODELS.keys()))
            raise ValueError(
                f"Unknown error_type '{self.error_type}'. Available error types: {available}."
            )
        return error_model_cls()

    def _mask_raw_data(self):
        if self.error_type == "berkson":
            self.cluster_based = True

        self.masked_data = self.raw_data.apply(
            lambda col: self.error_model.apply_to_column(self, col),
            axis=0,
        )

    def _assign_cluster(self, data, type="k-means", n_neighbors=100):
        """
        For cluster based error, fit the cluster on the original data. This will then be used to apply the error-structure.
        """
        if type != "k-means":
            raise ValueError(
                f"The prior cluster type is not 'k-means-constrained' [was {type} instead]"
            )
        working_df = data.select_dtypes(include="number")

        k_means = sklearn.cluster.KMeans(n_clusters=(self.n // n_neighbors) + 1)
        k_means.fit(np.array(working_df))
        return Cluster(fit=k_means, data=working_df)

    def drop_column(self, colname):
        self.raw_data = self.raw_data.drop(colname, axis=1)
        self.masked_data = self.masked_data.drop(colname, axis=1)

    def filter_cluster_raw(self, cluster):
        return self.raw_data[self.raw_filter == cluster]

    def viz_error_effect(self, varname, show=True, include_diagnostics=True):
        if varname not in self.raw_data.columns:
            raise ValueError(f"Column '{varname}' not found in data.")

        raw = self.raw_data[varname]
        masked = self.masked_data[varname]
        figures = []

        if raw.dtype.name == "category":
            fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
            raw.value_counts(dropna=False).plot(kind="bar", ax=ax[0], color="steelblue")
            ax[0].set_title("Raw")
            ax[0].set_xlabel(varname)
            ax[0].set_ylabel("Count")

            masked.value_counts(dropna=False).plot(kind="bar", ax=ax[1], color="indianred")
            ax[1].set_title(f"With error [{self.error_type}]")
            ax[1].set_xlabel(varname)

            fig.suptitle(f"Categorical error effect: {varname}")
            fig.tight_layout()
            figures.append(fig)
        elif np.issubdtype(raw.dtype, np.number):
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            raw.plot.kde(ax=ax[0], label="Raw", color="steelblue")
            masked.plot.kde(ax=ax[0], label="With error", linestyle="--", color="indianred")
            ax[0].set_title(f"KDE [{self.error_type}]")
            ax[0].set_xlabel(varname)
            ax[0].legend()

            diff = masked - raw
            ax[1].hist(diff, bins=30, color="slategray", alpha=0.8)
            ax[1].set_title("Error distribution (masked - raw)")
            ax[1].set_xlabel("Difference")
            ax[1].set_ylabel("Count")

            fig.suptitle(f"Numeric error effect: {varname}")
            fig.tight_layout()
            figures.append(fig)
        else:
            raise TypeError(
                f"Unsupported dtype for column '{varname}': {raw.dtype}. "
                "Only numeric and categorical columns are supported."
            )

        if self.error_type == "ePIT" and include_diagnostics and np.issubdtype(raw.dtype, np.number):
            s = self.raw_data[varname].dropna()
            x = s.values
            n = len(x)

            x_sorted = np.sort(x)
            F_emp = np.arange(1, n + 1) / (n + 1)
            u = s.rank(method="average").values / (n + 1)
            z = norm.ppf(u)

            fig_epit_1, axes = plt.subplots(2, 2, figsize=(10, 6))
            axes[0, 0].hist(u, bins=20, density=True, orientation="horizontal")
            axes[0, 0].set_title(r"hist of $\hat F_X(X)$")
            axes[0, 0].set_xlabel("u")
            axes[0, 0].set_ylabel("density")

            axes[0, 1].plot(x_sorted, F_emp)
            axes[0, 1].set_title(r"Empirical CDF $\hat F_X(x)$")
            axes[0, 1].set_xlabel("x")
            axes[0, 1].set_ylabel("F")

            s.plot.kde(ax=axes[1, 1])
            axes[1, 1].set_title(r"Empirical density $\hat f_X(x)$")
            axes[1, 1].set_xlabel("x")
            axes[1, 1].set_ylabel("density")
            axes[1, 0].axis("off")
            fig_epit_1.tight_layout()
            figures.append(fig_epit_1)

            fig_epit_2, axes2 = plt.subplots(2, 2, figsize=(10, 6))
            axes2[0, 0].hist(u, bins=20, density=True, orientation="horizontal")
            axes2[0, 0].set_title(r"hist of $U$ (PIT)")
            axes2[0, 0].set_xlabel("u")
            axes2[0, 0].set_ylabel("density")

            z_grid = np.linspace(-4, 4, 400)
            axes2[0, 1].plot(z_grid, norm.cdf(z_grid))
            axes2[0, 1].set_title(r"Standard normal CDF $\Phi(z)$")
            axes2[0, 1].set_xlabel("z")
            axes2[0, 1].set_ylabel("Phi(z)")

            axes2[1, 1].hist(z, bins=20, density=True, alpha=0.5, label="empirical Z")
            axes2[1, 1].plot(z_grid, norm.pdf(z_grid), label="N(0,1) pdf")
            axes2[1, 1].set_title(r"Standard normal density $\varphi(z)$")
            axes2[1, 1].set_xlabel("z")
            axes2[1, 1].set_ylabel("density")
            axes2[1, 1].legend()
            axes2[1, 0].axis("off")
            fig_epit_2.tight_layout()
            figures.append(fig_epit_2)

        if show:
            plt.show()
        return figures if len(figures) > 1 else figures[0]
