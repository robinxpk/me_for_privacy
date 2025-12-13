import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter

class CoxPHList: 
    def __init__(
            self, 
            # Data sets 
            true_data, 
            # variables
            duration_col, event_col, covariables = []
        ):
        self.data = true_data
        self.duration_col = duration_col
        self.event_col = event_col
        self.covariables = self.data.raw_data.columns
        if len(covariables) > 0: 
            self.covariables = covariables
        
        self.cph_ref = CoxPHFitter().fit(self.data.raw_data[self.covariables], duration_col = self.duration_col, event_col = self.event_col)

        self.fits = Fits()
        self.add_fit("reference", self.data)

    
    def add_fit(self, name, data):
        self.fits.add_new_fit(
            Fit(
                name, 
                data,
                CoxPHFitter().fit(data.masked_data[self.covariables], duration_col = self.duration_col, event_col = self.event_col), 
                self.cph_ref
            )
        )

class FitList: 
    # TODO: All fits should be inherited from this object to not be redundant.... 
    def __init__(self): 
        pass
# %%
class LMList():
    def __init__(self, true_data, formula):
        # Check how super works. Do not remember right now...
        # super.__init__(self)
        self.data = true_data
        self.formula = formula
        self.lm_ref = LM(data = self.data.raw_data, formula = self.formula)

        self.fits = Fits()
        self.add_fit("reference", self.data)

    def add_fit(self, name, data):
        self.fits.add_new_fit(
            Fit(
                name, 
                data,
                LM(data.masked_data, formula = self.formula), 
                self.lm_ref
            )
        )

# %%
class LM: 
    # Wrapper to make lm-fit quack like a duck (cph-fit)
    def __init__(self, data, formula): 
            self.data = data
            self.formula = formula
            self.fit = self._fit(self.data, self.formula)
            self.params_ = self.fit.params

    def _fit(self, data, form = "LBXT4 ~ RIDAGEYR + bmi + LBXTC"): 
        return smf.ols(form, data=data).fit()


class Fits: 
    def __init__(self):
        self.fits = {}

    def __getattr__(self, name): 
        # called only if normal attribute lookup fails
        try:
            return self.fits[name]
        except KeyError:
            raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def add_new_fit(self, Fit): 
        self.fits[Fit.name]= Fit
    
    def table_all_estimates(
            self, table_name, table_as_markdown = False,
            # Allow to define a getter function to be able to extract different summary statistics: 
            getter = lambda mdl_fit: mdl_fit.fit.params_
            # or e.g.: 
            # getter=lambda mdl_fit: mdl_fit.fit.standard_errors_
            # getter=lambda mdl_fit: mdl_fit.fit.summary["p"]
        ): 
        print(table_name)
        table = pd.concat(
            [df.rename(name) for name, df in zip(self.fits.keys(), [getter(mdl_fit) for mdl_fit in self.fits.values()])],
            axis=1,
        )

        if table_as_markdown: 
            return table.to_markdown()
        return table

    def boxplot_all_estimates(
        self, plot_name, y_lab = "Estimate", marker_in_boxplot = "reference", dot_color = "r",
        # Allow to define a getter function to be able to extract different summary statistics: 
        getter = lambda mdl_fit: mdl_fit.fit.params_
        # or e.g.: 
        # getter=lambda mdl_fit: mdl_fit.fit.standard_errors_
        # getter=lambda mdl_fit: mdl_fit.fit.summary["p"]
    ): 
        table = pd.concat(
            [df.rename(name) for name, df in zip(self.fits.keys(), [getter(mdl_fit) for mdl_fit in self.fits.values()])],
            axis=1,
        )
        plt_table = table.transpose()
        
        fig, ax = plt.subplots(figsize=(12, 4))
        plt_table.boxplot(ax = ax)
        ax.scatter(np.arange(1, len(table) + 1), plt_table.loc[marker_in_boxplot, :], color=dot_color, zorder=3)

        plt.title(plot_name)
        plt.ylabel(y_lab)
        ax.tick_params(axis = "x", rotation=90)
        plt.tight_layout()
        plt.show()

class Fit: 
    def __init__(self, name, data, fitted_mdl, ref_mdl):
        self.name = name
        self.data = data
        self.fit = fitted_mdl
        self.ref_fit = ref_mdl
        self.bias = self.fit.params_ - self.ref_fit.params_
