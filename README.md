#### TODOs ####
- [ ] Raphael hat mit masked arrays oder so gearbeitet. Die Matritzen kombinieren die Information über Fehlerspalten und Werte. Klingt besser. 
- [ ] Potentially change sampler to not be forced to have derivatives; especially for ePIT: Could use eCDF and would not need invertible and continuous proxi of CDF (using sigmoid as of now).
- [ ] lognormal density assumes single error variance, i.e. only one column touched by error; Extend logic to work with vectors.
- [ ] Posterior as object? Can create one parent an inherit from that because likelihood and all (so far) is the same among error types. 
- [ ] Read Literature on previous approaches for statistical disclosure control using measurement error;i.e. read ME papers (next week).
- [ ] Prepare presentation (next week).
- [ ] Error on all variables (next week).
- [ ] Clean up Data class (later).
- [ ] Prior (and Posterior) predictive checks (next week?).
- [ ] Dealing with outliers: Distanz zum Datenmittelpunkt als Gewicht für die Fehlertermvarianz? (Outlook für die verschiedenen Fehlerterme)
- [ ] Burn-in vs warmup [read this](https://statmodeling.stat.columbia.edu/2017/12/15/burn-vs-warm-iterative-simulation-algorithms/).
- [ ] Densities nicht ganz korrekt wegen Intercept und shape von beta; hier nochmal überarbeiten (Siehe generelles Setup und Additive).

- [ ] Implement varying degrees of error **including** the one error equivalent to Berkson.
- [ ] Mit Helena über Präsentation reden.
- [ ] Umstellen der Daten auf full nhanes, nicht den alten mit nur $400$ (See GitHub Repo for Code to read in the data); WENN ZEIT.
- [ ] Berkson Error: Hat die Clusterzurodnung (pro Variable vs. Clusterbildung über alle Variablen) einen Effekt auf die Fehlerstruktur bzw. gibt es hier falsch / richtig? (Nochmal mit Helena besprechen, was hier gemeint war) 

- [ ] **Note:** Allow for variable $j$ specific error variance $\sigma^{2}_{\epsilon, j}$!!, i.e. within a variable, the error variance is constant, but among variables, the error variance may differ.

--- 
Now

- [ ] Data Object is so bad, wtf... Fix this.
- [ ] Clarify Names in Class and Code!
- [ ] Parallelisieren.
- [ ] Posteriors ausformulieren.

- [ ] Use all Bayesian Models (no freq as reference).
- [ ] Write note on how error variance works (i.e. on not identifiable).
    - [ ] Write section on attenuation factor and why (thus) the measurement error variance has to be specified. (Already referenced it somewhere. lol.)
        - [ ] Mention that, due to fixing the error variance, the model does not over-compensate.
    - [ ] Formulate section where I go into *why* this approach works and how it accounts for the high dimensionality. What is the frequentist equivalent to this?
- [ ] The structure of the ErrorModel object in the Data object can be greatly simplified now that the error (for sure) applies to the full column.

# Measurement Error Synthetic Data
## Virtual Environment Setup (Conda)

Based on the given `.yml`-file, create a conda environment using: 

`conda env create -f environment.yml`

In case one wants to update the `.yml` file, create a new file using: 

`conda env export --from-history > environment.yml` 

Some useful prompts: 

- list environments: `conda info --envs`
- Remove environment and its dependencies: `conda create --name envname`

!! When running code in VSCode, check that the correct python interpreter is used: 

1) cntrl + p --> Type "Python: Select Interpreter" --> Select correct environment

2) When using interactive mode, check that environment is used (displayed on the top right corner)

## Data Sets

### National Health and Nutrition Examination Survey [NHANES]

United States’ main survey for getting nationally representative data on the actual health and nutritional status of people living in the U.S. — not just what they say in a questionnaire. 

Unique because it combines: 
1. Interviews (at home): demographics, income, health history, health behaviors, diet
2. Standardized physical exams (in mobile exam centers)
3. Laboratory tests (blood, urine, sometimes environmental or infectious markers)

- Target population: civilian, noninstitutionalized U.S. population. Certain groups (older adults, some minority groups) are oversampled so you can make precise estimates for them.
- Design: complex, multistage, stratified probability sample → you must use the supplied survey weights, strata, and PSU variables when doing analysis, or your estimates will be biased and SEs too small. That’s why there’s a whole “analytic guidelines” page
- Data files split into modules: Demographics, Dietary, Examination, Laboratory, Questionnaire

#### Download
[nhanes](https://www.cdc.gov/nchs/nhanes/index.html)

Focused on "Demographic Variables and Sample Weights" from August 2021-August 2023 for now, downloaded [here](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023).
Find Doc-Files after selecting the data set of choice.

Also, for Cox-models fitted during brainstorming, we used the data from [this](https://doi.org/10.1093/ije/dyaa164) paper. Its github can be found [here](https://github.com/chiragjp/voe) (file: `nhanes9904_VoE.Rdata`).

Dietary1 dataset: [here](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Dietary&Cycle=2021-2023)

Body Measures: [here](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=2021-2023)

### The third International Stroke Trial [IST-3]
[IST-3](https://datashare.ed.ac.uk/handle/10283/1931)

## Modelle ##
### Reference Model ###
Here, $x_{j}$ refers to observed("true") values. I.e. without introducing any ME.
1. Likelihood 
$$
\begin{aligned}
    y_{i, lbxt4} \mid \boldsymbol{\beta}, \sigma^2 &\sim N(\mu_i, \sigma^2)\\
    \mu_i &= \beta_0 + \beta_{age} x_{i, age} + \beta_{bmi} x_{i, bmi} + \beta_{kcal} x_{i, kcal}
\end{aligned}
$$
2. Priors
$$
\begin{aligned}
    \beta_j &\overset{iid.}{\sim} N(0, b^2) \text{ with given }b \\
    \sigma^2 &\sim Ga(c, d) \text{ with given }c, d
\end{aligned}
$$

Posterior: 
$$
\begin{aligned}
    \log(p(\beta, \sigma^2\mid \boldsymbol{y})) &\propto -\frac{n}{2}log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}{(y_i - \boldsymbol{x}_i^T \boldsymbol{\beta})^2} \\
    &- \frac{1}{2b^2} \sum_{j=1}^{p}{\beta_{j}^2} + (c-1) \log(\sigma^2) - d \sigma^2 + C
\end{aligned} 
$$

### Naive Model ###
Model to quantify the effect of the measurement error correction: Same as above, but use the erroneous observations:
$$
\begin{aligned}
    y_{i, lbxt4} &\sim N(\mu_i, \sigma^2)\\
    \mu_i &= \beta_0 + \beta_{age} \tilde{x}_{i, age} + \beta_{bmi} \tilde{x}_{i, bmi} + \beta_{kcal} \tilde{x}_{i, kcal}
\end{aligned}
$$
Rest as above, but with $\tilde{x}_{ij}$.

### ME correction ###
We correct for the measurement error using latent variables and separation into different sub-models. 
#### A) Linear Model ####
Simple linear model (Connects exposure und covariates, $p(y|x)$): 
$$
\begin{aligned}
    y_{lbxt4} &\sim N(\mu, \sigma^2)\\
    \mu &= \beta_0+ \beta_{age} x_{age} + \beta_{bmi} x_{bmi} + \beta_{kcal} x_{kcal} \\

\end{aligned}
$$
Priors
$$
\begin{aligned}
    \beta_j &\overset{iid.}{\sim} N(0, b^2) \text{ with given }b \\
    \sigma^2 &\sim Ga(c, d) \text{ with given }c, d
\end{aligned}
$$
Same all linear models above. Note that this model uses the *true* values of the covariates $x_{ij}$. 
#### B) Measurement Model ####
This model specifies how the measurement error is applied to the true values. 
**Important:** The (true) value of the error variance $\sigma^2_{\epsilon}$ must be supplied to the sampler. Else, the problem is not identifiable (See [[#Attenuation Factor]])! 

e.g.: Gaussian, additive error:
$$
\begin{aligned}
    \tilde{x}_{ij} &= x_{ij} + \epsilon_{ij} \text{ where } \epsilon \sim N(0, \sigma^2_{\epsilon}), 
\end{aligned}
$$
where $\sigma^2_{\epsilon}$ is known.

#### C) Exposure Model ####
Finally, the exposure models specifies the distribution of the true values themselves, i.e. distributional assumptions on latent (true) variable $x$. 
$$
\begin{aligned}
    x_{ij} \sim F_p
\end{aligned}
$$

**Important.** Assume a reasonable distribution $F_p$ here. Else, the corrected values may be biased, too. 
For now, we can use the empirical density of the original variable (where "original" refers to the variable without error); </br> 
TODO: This might be too much information published, but maybe not? 
- Can supply empirical distribution as function to JAX. 
- How problematic is a publication of the error-free data distribution? 

#### Likelihood ####
$$
\begin{aligned}
    p(\boldsymbol{\beta}, \boldsymbol{X}, \sigma| y, \boldsymbol{\tilde{X}}, \sigma^2_{\epsilon}) \propto p(\sigma^2)p(\boldsymbol{\beta})\prod_{i=1}^{n}{p(y_i|\boldsymbol{\beta}, \boldsymbol{x}_i, \sigma^2) p(\boldsymbol{\tilde{x_i}}|\boldsymbol{x_i}, \sigma_\epsilon^2)}p(\boldsymbol{x_i})
\end{aligned}
$$

Für priors: 
1. Normalverteilung mit hoher Varianz 
2. Simulation 

! Prior predictive checks


## Errors
*Note: An error should only increase the variance and not introduce any bias!* 

This section presents the errors we added to the data. For each error, I give
1. Mathematical Definition
2. Specification of the BHM
3. (Unnormalized) Posterior

The latter is used to implement the `log_posterior`-function in `python`.

In the following, we assume a total of $K$ variables where $j \in \{ 1, ..., K\}$ denotes the $j$-th variable. Each variable is observed $n$ times with $i = 1, ..., n$ denoting the $i$-th observation.

#### Evaluation of the Error-Degree ####
The uncertainty introduced by each error is evaluated using the uncertainty evaluation formula (UEF) defined variable-wise as 
$$
\begin{aligned}
    nMSE_j = \frac{\frac{1}{n}(\boldsymbol{x}_{j, true} - \boldsymbol{x}_{j, error})^T(\boldsymbol{x}_{j, true} - \boldsymbol{x}_{j, error})}{\widehat{Var}(x_{j, true})}
\end{aligned}
$$

We aim to have each error introduce the same level of uncertainty within each variable $j \in \{1, ..., K\}$ such that results are comparable among error types. 

### Additive normal error 
To each variable, we just add a normally distributed random variable with an expected value of $0$. 
The variance of the normal distribution affects the UEF. 

#### Mathematical definition ####
$$
\begin{aligned}
\begin{split}
    \tilde{x}_{ji} = x_{ji} + \epsilon_{ji} \\
    \text{where } \epsilon_{ji} \overset{indep.}{\sim} N(0, \sigma^2_{\epsilon, j}) 
\end{split}
\end{aligned}
$$
**Note:** Allow for variable $j$ specific error variance $\sigma^{2}_{\epsilon, j}$!!, i.e. within a variable, the error variance is constant, but among variables, the error variance may differ
(Thus, not iid., but only independently distributed).

#### Model Specification ####
**A) Linear Model**. 

Likelihood. 
$$
\begin{aligned}
    y_{i, lbxt4} &\sim N(\mu_i, \sigma^2)\\
    \mu_i &= \beta_0 + \beta_{age} x_{i, age} + \beta_{bmi} x_{i, bmi} + \beta_{kcal} x_{i, kcal} \\
\end{aligned}
$$
Priors. 
$$
\begin{aligned}
    \boldsymbol{\beta} &\sim  N(\boldsymbol{0}, b^2 \boldsymbol{I}_{p \times p}) \text{ with given }b, \\
    \sigma^2 &\sim Ga(c, d) \text{ with given }c, d. 
\end{aligned}
$$

**B) Measurement Model.** </br> 
As mentioned above, the measurement model requires the error variance to be known for identification purposes. 
$$
\begin{aligned}
    \boldsymbol{\tilde{x}}_{i}\mid \boldsymbol{x}_i, \boldsymbol{\sigma}_{\epsilon}^2& \sim N\left(  
\boldsymbol{x}_i := \left(  
\begin{matrix}
    x_{i, age} \\
    x_{i, bmi} \\
    x_{i, kcal}
\end{matrix}
\right), 
\boldsymbol{G} := \left(  
\begin{matrix}
\sigma^2_{\epsilon, age} & 0 & 0 \\
0 & \sigma^2_{\epsilon, bmi} & 0 \\
0 & 0 & \sigma^2_{\epsilon, kcal} 
\end{matrix}
\right)
\right)
\end{aligned}
$$
Where all $\sigma^2_{\epsilon, j}$ are given, i.e. **no priors!** Which nicely simplifies the (log-)posterior. 

**C) Exposure Model.**
$$
\begin{aligned}
    \boldsymbol{x}_{i} \sim f_p
\end{aligned}
$$
where $f_p$ is the joint empirical density of $X_{age}, X_{bmi}, X_{kcal}$ which is evaluated at $x_{i, age}, x_{i, bmi}, x_{i, kcal}$ in the unnormalized posterior likelihood. In `JAX`, this is achieved by passing a callable and **JAX compatible** function to the log-density. 

#### Unnormalized Posterior ####
The following unnormalized posterior assumes the whole $\boldsymbol{x}_i$ vector is affected by the error. 
If this would not be the case, the density prior $p(\boldsymbol{\tilde{x}}_i\mid \boldsymbol{x_i})$ simplifies: 
It then only refers to the variables $j$ that are affected by the error. 
This implies that the shape of $\boldsymbol{G}$ changes, too! 
(e.g. Just implemented code where only the column of `kcal` has an additive error. Then, $\boldsymbol{G}$ is just a scalar equal to the error variance).
$$
\begin{aligned}
    &p(\boldsymbol{X}, \boldsymbol{\beta}, \sigma^2 \mid \boldsymbol{\tilde{X}}, \boldsymbol{y}, \boldsymbol{G})\\
   &\propto
\left( 
\prod_{i=1}^{n}{
    p(y_{i}\mid \boldsymbol{x}_{i}, \boldsymbol{\beta}, \sigma^2)
    p(\boldsymbol{\tilde{x}}_i \mid \boldsymbol{x}_i)
    p(\boldsymbol{x}_i)
}
\right)
p(\boldsymbol{\beta})
p(\sigma^2)
\\

&= 
\left( 
\prod_{i=1}^{n}{
     \left[N(\boldsymbol{x}^T_{i}\boldsymbol{\beta}, \sigma^2)\right]
   \left[N(\boldsymbol{x}_i, \boldsymbol{G})\right]
   \left[f_p(\boldsymbol{x}_i)\right]
}
\right)
N(\boldsymbol{0}, b^2 \boldsymbol{I}_{p \times p})
Ga(c, d)
\end{aligned}
$$
Define $\boldsymbol{z}_i = (1, \boldsymbol{x}_i)$ such that
$$
\begin{align*}
\propto &
\left(  (\sigma^2)^{-n / 2}\prod_{i=1}^{n}{
\left[  
    \exp\left( -\frac{1}{2\sigma^2} (y_i - \boldsymbol{z}_i^T\boldsymbol{\beta})^2 \right)
\right]
\left[  
    \exp\left( -\frac{1}{2} (\boldsymbol{\tilde{x}}_i - \boldsymbol{x}_i   )^T \boldsymbol{G}^{-1}(\boldsymbol{\tilde{x}}_i - \boldsymbol{x}_i ) \right)
\right] 
\left[ f_p(\boldsymbol{x}_i) \right]
}
\right)\\
& \times \det(b^2 \boldsymbol{I}_{p \times p})^{-1 / 2} \exp\left( - \frac{1}{2b^2} \boldsymbol{\beta}^T \boldsymbol{\beta} \right) (\sigma^2)^{c-1} \exp\left( -d \sigma^2 \right) \\
\end{align*}
$$

The log posterior is 
$$
\begin{align*}
     \log & [p(\boldsymbol{x}, \boldsymbol{\beta}, \sigma^2 \mid \boldsymbol{\tilde{x}}, \boldsymbol{y}, \boldsymbol{G})]\\
    & \propto -\frac{n}{2} \log \sigma^2 \\
    & + \sum_{i=1}^{n}{\left( -\frac{1}{2\sigma^2}(y_i - \boldsymbol{z}_i^T \boldsymbol{\beta})^2 - \frac{1}{2} (\boldsymbol{\tilde{x}}_i - \boldsymbol{x}_i)^T \boldsymbol{G}^{-1}(\boldsymbol{\tilde{x}}_i - \boldsymbol{x}_i) + \log f_p(\boldsymbol{x}_i)\right)} \\
    & - \frac{p}{2} \log b^2 - \frac{1}{2b^2} \boldsymbol{\beta}^T \boldsymbol{\beta} \\
    & + (c-1) \log \sigma^2 - d \sigma^2 \\
\end{align*}
$$
To sample from real line, express the posterior in terms of $\log \sigma^2 =: \upsilon$ bzw. $\log \sigma^2_{\epsilon, j} =: \upsilon_{\epsilon, j}$.  
Use change of variable and simply add the $\log |J|$ for each transformation (Note: Only relevant for the densities related to the variances, but express full density in terms of log variance). 
$$
\begin{align*}
     \log & [p(\boldsymbol{x}, \boldsymbol{\beta}, \upsilon\mid \boldsymbol{\tilde{x}}, \boldsymbol{y}, \boldsymbol{G})]\\
    & \propto -\frac{n}{2} \upsilon \\
    & + \sum_{i=1}^{n}{\left( -\frac{1}{2 \exp \upsilon}(y_i - \boldsymbol{z}_i^T \boldsymbol{\beta})^2 - \frac{1}{2} (\boldsymbol{\tilde{x}}_i - \boldsymbol{x}_i)^T G^{-1}(\boldsymbol{\tilde{x}}_i - \boldsymbol{x}_i) + \log f_p(\boldsymbol{x}_i)\right)} \\
    & - \frac{p}{2} \log b^2 - \frac{1}{2b^2} \boldsymbol{\beta}^T \boldsymbol{\beta} \\
    & + (c-1) \upsilon - d \exp \upsilon  \underbrace{+ \upsilon}_{+ \log |J|}\\
\end{align*}
$$

### Multiplicative log-normal error


Every variable is multiplied by a log-normally distributed random variable. 

All that changes compared to the [[#Additive normal error]] is the *Measurement Model*. For sake of completeness, I repeat specify the full model and the full posterior, but the difference is in the measurement model only! 

#### Mathematical definition of log-normal error
$$
\begin{aligned}
    \tilde{x}_{ij} &= x_{ij} \cdot \epsilon_{ij} \\
    &\text{where } \log(\epsilon_{ij}) \overset{iid.}{\sim}N\left(\mu_{(log)} , \sigma_{(log)}^2\right) \\ 
    &\leftrightarrow \epsilon_{ij} \overset{iid.}{\sim} \text{Lognormal}\left(\exp\left[ \mu_{(log)}+ \frac{\sigma_{(log)}^2}{2} \right] , \left(\exp[\sigma^2_{(log)}]-1\right)\exp\left[ 2 \mu_{(log)}+\sigma^2_{(log)} \right] \right).
\end{aligned}
$$

To have $\mathbb{E}_{ }\left[ \tilde{x}_{ij}\right] = x_{ij}$, choose $\mathbb{E}_{}\left[ \epsilon_{ij}\right] = 1$, i.e. $\mu_{(log)}= -\frac{\sigma^2_{(log)}}{2}$. 

This implies the simplification to
$$
\begin{aligned}
    \epsilon_{ij} \overset{iid.}{\sim} \text{Lognormal}\left( 1,  \sigma^2_{\epsilon, j}\right), 
\end{aligned}
$$
where $\sigma^2_{\epsilon}:=\exp\left( \sigma^2_{(log)} \right)-1$.

This gives
$$
\begin{aligned}
    \tilde{x}_{ij} \mid x_{ij} \sim \text{Lognormal}\left( x_{ij}, \sigma^2_{\epsilon, j}x_{ij}^2 \right).
\end{aligned}
$$
$\rightarrow$ The error does not introduce bias

**Important note for the posterior:** The lognormal pdf is expressed in terms of its *logarithmic* location and scale values. In other words, use this density with the $\log$ scale location- and scale-parameters $\mu_{(log)}$  and $\sigma^2_{(log)}$:

$$
\begin{aligned}
    p(\epsilon\mid \mu_{(log)}, \sigma^2_{(log)}) = 
\frac{1}{\epsilon \cdot \sqrt[]{2\pi\sigma_{(log)}^2}} \exp\left( - \frac{(\log \epsilon - \mu_{(log)})^2}{2\sigma^2_{(log)}} \right)
\end{aligned}
$$
bzw. due to $\log \tilde{x} \mid x = \log x + \log \epsilon \sim N(\log x + \mu_{(log), j}, \sigma^2_{(log), j})$: 
$$
\begin{aligned}
    p(\tilde{x}\mid x, \mu_{(log)}, \sigma^2_{(log)}) = 
\frac{1}{\tilde{x}\cdot \sqrt[]{2\pi\sigma_{(log)}^2}} \exp\left( - \frac{(\log \tilde{x}- [\log x + \mu_{(log))}])^2}{2\sigma^2_{(log)}} \right)
\end{aligned}
$$

$\rightarrow$ While the above expresses $\epsilon$ on its original scale to proof "no bias", the density of $\epsilon$ is used in the (log-)posterior.

**!! Because we take the $\log$** of the observed values $x$ bzw. $\tilde{x}$, these must be strictly positive! 

#### Model Specification ####
**A) Linear Model**. 
Likelihood
$$
\begin{aligned}
    y_{i, lbxt4} &\sim N(\mu_i, \sigma^2)\\
    \mu_i &= \beta_0 + \beta_{age} x_{i, age} + \beta_{bmi} x_{i, bmi} + \beta_{kcal} x_{i, kcal},
\end{aligned}
$$
with
$$
\begin{aligned}
    \boldsymbol{\beta} &\sim  N(\boldsymbol{0}, b^2 \boldsymbol{I}_{p \times p}) \text{ with given }b, \\
    \sigma^2 &\sim Ga(c, d) \text{ with given }c, d. 
\end{aligned}
$$

**B) Measurement Model.** </br> 
$$
\begin{aligned}
    \tilde{x}_{ij} \mid x_{ij} \sim \text{Lognormal}\left(x_{ij}, \sigma^2_{\epsilon, j}x_{ij}^2 \right), 
\end{aligned}
$$
As above: To be identifiable, I need to specify the measurement error variance $\sigma^2_{(log)}$. 

**C) Exposure Model.**
$$
\begin{aligned}
    \boldsymbol{x}_{i} \sim f_p
\end{aligned}
$$

#### Unnormalized Posterior ####
As above, I assume that the whole covariate-vector $\boldsymbol{x}_j$ is affected by the error. See [[#Additive normal error]] for more information.

Additionally, this derivation is based on the independence between error terms to specify the joint density of $p(\tilde{x}_{ij} \mid x_{ij})$: 
For independent error terms, the conditional distributions $p(\tilde{x}_{ij} \mid x_{ij})$ are independent for all observations $i = 1, ..., n$ and all covariates $j = 1, ..., p$ $\rightarrow$ build product. 

**Note:** Only need to adjust $p(\tilde{x}_{ij}\mid x_{ij})$ compared to above. But be aware that the location and scale parameters are on the log-scale of the error (See mathematical definition of lognormal error), i.e.
$\mu_{(log), j} = -\frac{\sigma^2_{(log), j}}{2}$. 

$$
\begin{aligned}
    &p(\boldsymbol{X}, \boldsymbol{\beta}, \sigma^2 \mid \boldsymbol{\tilde{X}}, \boldsymbol{y}, \boldsymbol{G})\\
   &\propto
\left( 
\prod_{i=1}^{n}{
    p(y_{i}\mid \boldsymbol{x}_{i}, \boldsymbol{\beta}, \sigma^2)
    p(\boldsymbol{\tilde{x}}_i \mid \boldsymbol{x}_i)
    p(\boldsymbol{x}_i)
}
\right)
p(\boldsymbol{\beta})
p(\sigma^2)
\\

&= 
\left( 
\prod_{i=1}^{n}{
     \left[N(\boldsymbol{x}^T_{i}\boldsymbol{\beta}, \sigma^2)\right]
   \left[logNorm(\boldsymbol{x}_i, \boldsymbol{G})\right]
   \left[f_p(\boldsymbol{x}_i)\right]
}
\right)
N(\boldsymbol{0}, b^2 \boldsymbol{I}_{p \times p})
Ga(c, d)
\end{aligned}
$$
Define $\boldsymbol{z}_i = (1, \boldsymbol{x}_i)$ such that
$$
\begin{align*}
\propto &
\left(  (\sigma^2)^{-n / 2}\prod_{i=1}^{n}{
\left[  
    \exp\left( -\frac{1}{2\sigma^2} (y_i - \boldsymbol{z}_i^T\boldsymbol{\beta})^2 \right)
\right]
\left[  
    \prod_{j=1}^{p}\frac{1}{\tilde{x}_{ij}\sqrt[]{\sigma_{(log),j}^2}}\exp\left(-\frac{1}{2\sigma_{(log),j}^2}(\log\tilde{x}_{ij}-[\log x_{ij} + \mu_{(log), j}])^2\right)
\right] 
\left[ f_p(\boldsymbol{x}_i) \right]
}
\right)\\
& \times \det(b^2 \boldsymbol{I}_{p \times p})^{-1 / 2} \exp\left( - \frac{1}{2b^2} \boldsymbol{\beta}^T \boldsymbol{\beta} \right) (\sigma^2)^{c-1} \exp\left( -d \sigma^2 \right) \\
\end{align*}
$$

The log posterior is 
$$
\begin{align*}
     \log & [p(\boldsymbol{x}, \boldsymbol{\beta}, \sigma^2 \mid \boldsymbol{\tilde{x}}, \boldsymbol{y}, \boldsymbol{G})]\\
    & \propto -\frac{n}{2} \log \sigma^2 \\
    & + \sum_{i=1}^{n}{\left( -\frac{1}{2\sigma^2}(y_i - \boldsymbol{z}_i^T \boldsymbol{\beta})^2 + 
\sum_{j=1}^{p} \left( - \frac{1}{2}\log\sigma_{(log),j}^2-\log\tilde{x}_{ij}-\frac{1}{2\sigma_{(log),j}^2}\left(\log\tilde{x}_{ij} - \log x_{ij} + \sigma_{(log), j}^2 / 2\right)^2\right) + \log f_p(\boldsymbol{x}_i)\right)} \\
    & - \frac{p}{2} \log b^2 - \frac{1}{2b^2} \boldsymbol{\beta}^T \boldsymbol{\beta} \\
    & + (c-1) \log \sigma^2 - d \sigma^2 \\
\end{align*}
$$
To sample from real line, express the posterior in terms of $\log \sigma^2 =: \upsilon$ bzw. $\log \sigma^2_{\epsilon, j} =: \upsilon_{\epsilon, j}$.  
Use change of variable and simply add the $\log |J|$ for each transformation (Note: Only relevant for the densities related to the variances, but express full density in terms of log variance). 
$$
\begin{align*}
     \log & [p(\boldsymbol{x}, \boldsymbol{\beta}, \upsilon\mid \boldsymbol{\tilde{x}}, \boldsymbol{y}, \boldsymbol{G})]\\
    & \propto -\frac{n}{2} \upsilon \\
    & + \sum_{i=1}^{n}{\left( -\frac{1}{2 \exp \upsilon}(y_i - \boldsymbol{z}_i^T \boldsymbol{\beta})^2 + 
\sum_{j=1}^{p} 
\left( - \frac{1}{2}\log\sigma_{(log),j}^2-\log\tilde{x}_{ij}-\frac{(\log \tilde{x}_{ij} - \log x_{ij} + \frac{1}{2}\sigma^2_{(log), j})^2}{2\sigma^2_{(log), j}}\right) + \log f_p(\boldsymbol{x}_i)\right)} \\
    & - \frac{p}{2} \log b^2 - \frac{1}{2b^2} \boldsymbol{\beta}^T \boldsymbol{\beta} \\
    & + (c-1) \upsilon - d \exp \upsilon  \underbrace{+ \upsilon}_{+ \log |J|}\\
\end{align*}
$$

### ePIT error
- Could I avoid sharing the true distribution of the variables $x$ and instead share the distribution of variables $z$? Then the information about the empirical density itself is contained in the transformation, but one cannot backtrack to the original values? Because that would require the actual empirical density? 

This error is based on the empirical probability integral transform (ePIT) bzw. the empirical CDF function. @okhrinBasicElemts2017, p. 195 defines this as 
$$
\begin{aligned}
    \hat{F}_{j}(x) = \frac{1}{n+1} \sum_{k=1}^{n}{I(x_{jk} \leq x)}.
\end{aligned}
$$
*Note: $\hat{F}_{j}$ is **not** continuous! For every $x \in (x_{j(k)}, x_{j(k+1)})$, it is constant ($x_{j(k)}$ denotes the $k$-th ordered observation.)*
Below, we thought of a continuous proxi in case we are interested in derivates.

For each $x_{ji} \in \{x_{j1}, ..., x_{jn}\}$, we define $p_{ji} := \hat{F}_{j}(x_{ji})$. Due to the properties of the (empirical) CDF, we know that $p_{ji} \overset{iid.}{\sim} \text{Uniform}(0, 1)$. Note that the empirical CDF basically assigns normalized ranks. Thus, the following uses the term "rank" somewhat interchangeable with the empirical CDF value as it is pretty much the same. </br> 
From here, we can add any type of error from any distribution by: 
1) transforming $p_{ij}$ using a quantile function of any distribution which would yield $Q_{}(p_{ij})$ to then follow this distribution where $Q$ denotes the quantile function of this chosen distribution
2) We add an error onto $Q(p_{ij})$ of any kind, i.e. of any distribution
3) We retransform $Q(p_{ij})$ using the *known* CDF bzw. inverse of $Q$ to again obtain a noisy $p_{ij}$. 
4) From noisy $p_{ij}$ we can then go back to the original scale using $\hat{F}^{-1}_{j}$ from the above equation.

Of course, we could skip steps if we would just add noise on the uniform variable $p_{ij}$. Maybe this would be more efficient and / or easier to model? 

For now, we used the std. normal quantile function, added a std. normal error and then re-transformed the variable to its original scale using the empirical CDF. 

Overall, this approach has the upside of never introducing new values. Instead, all noisy data points are still real data points. 
Essentially: </br>
This type of error build rank-value pairs. Then takes every observation and potentially (that is, most likely) assigns it a new rank. 
When all observations have been assigned a new rank, each rank is the value assigned which was saved in the rank-value pair. 
Rephrasing it like this seems to simplify things: It basically is an error on the assigned rank. 

#### Mathematical Definition ####
Observing $x_{ij}$, apply
$$
\begin{aligned}
\begin{split}
    (1)\ & p_{ji} = \hat{F}_{j}(x_{ji})\\
    (2)\ & z_{ji} = \Phi^{-1}(p_{ji}) \\
    (3)\ & \tilde{z}_{ji} = z_{ji} + \epsilon_{ij}  \text{ where } \epsilon_{ij} \overset{iid.}{\sim} N(0, \sigma^2_{\epsilon, j})\\
    (4)\ & \tilde{p}_{ji} = \Phi(\tilde{z}_{ji}) \\
    (5) \ & \tilde{x}_{ji} = \hat{F}^{-1}_{j}(\tilde{p}_{ji})
\end{split}
\end{aligned}
$$

In short: 
$$
\begin{aligned}
    \tilde{x}_{ji} = \hat{F}^{-1}_{j}(
        \Phi\{ 
            \Phi^{-1}[
                \hat{F}_{j}(x_{ji})
            ] + \epsilon_{ij}
       \} 
    )
\end{aligned}
$$

This expression can be generalized, I think: 
$$
\begin{aligned}
    \tilde{x}_{ji} = \hat{F}^{-1}_{j}(
        G(
                \hat{F}_{j}(x_{ij})
        )
    )
\end{aligned}
$$
where $G(.)$ is just any function that (potentially) assigns a new rank to $x_{ji}$ and $\hat{F}$ (re-)transform from rank to (observed) value. 
In the above case, 

$$
\begin{aligned}
    G(y) = \Phi(\Phi^{-1}(y) + \epsilon) \text{ where } \epsilon \overset{iid.}{\sim}N(0, \sigma_{\epsilon, j}^2)
\end{aligned}
$$

*Note: I expect that, depending on the added error, the distribution of $\tilde{x}_{ji}$ flattens compared to the original distribution. But because values cannot exceed the largest observed value, I expect a peaky behaviour towards the edges. Let's see!* :)

#### Model Specification ####
I think for this error, it might be easier to express the density in terms of $z_{ij}$ bzw. $\tilde{z}_{ij}$: 

Just directly use transformed sample values $z_{ij}$.
Reason is that 
1. I know the distribution of $z_{ij}$. For $x_{ij}$, I would need to apply *Change of Variable*.
2. $z_{ij}$ lies on the real line while $x_{ij}$ might not. As NUTS samples from real line, $z_{ij}$ seems a better choice. 

But I still want the $\beta$s to measure the effect of $x_{j}$. 
Should be no problem if I express the uncertainties in terms of $z$ and when conditioning on $x$ bzw. $z$, it is given and I may apply a transformation to obtain $x$ given $z$ which is not random any more. 
But all priors and the measurement error model is expressed in terms of $z$. 

[This STAN Guide](https://mc-stan.org/docs/stan-users-guide/reparameterization.html?utm_source=chatgpt.com) addresses such a modelling approach: </br> 
A transformation samples a parameter (here: $z$) and **then** transforms it (here: $x(z)$. 
Change of variable transforms a parameter and **then** samples it! And only change of variable requires the Jacobian term! </br> 
In other words: 
- If I want to change the distribution I sample from, e.g. as I do for the variance and log-variance to sample from the real line, I deal with change of variable $\rightarrow$ I assign a distribution to another scale of the variance. 
- If I want to use a sampled variable and transform it **without** changing the distribution I sample from, it's a simple transformation. 

Thus, I can express my linear model in terms of $x(z)$ to still draw comparable $\beta$s to before while evaluating the (log-)posterior in terms of $z$ bzw. $\tilde{z}$. 

**A) Linear Model**. 
Likelihood
$$
\begin{aligned}
    y_{i, lbxt4} &\sim N(\mu_i, \sigma^2)\\
    \mu_i &= \beta_0 + \beta_{age} x_{age}(z_{i, age}) + \beta_{bmi} x_{bmi}(z_{i, bmi}) + \beta_{kcal} x_{kcal}(z_{i, kcal}),
\end{aligned}
$$
where 
$$
\begin{aligned}
    x_{ij} := x_{j}(z_{ij}) = \hat{F}_{j}(\Phi(z_{ij})) \\
\end{aligned}
$$
which is based on the first two steps from above:
$$
\begin{aligned}
    (1)\ & p_{ji} = \hat{F}_{j}(x_{ji})\\
    (2)\ & z_{ji} = \Phi^{-1}(p_{ji}).
\end{aligned}
$$
Also, as before
$$
\begin{aligned}
    \boldsymbol{\beta} &\sim  N(\boldsymbol{0}, b^2 \boldsymbol{I}_{p \times p}) \text{ with given }b, \\
    \sigma^2 &\sim Ga(c, d) \text{ with given }c, d. 
\end{aligned}
$$

**B) Measurement Model.** </br> 
This model is expressed in terms of $\tilde{z}_{ij}$ on step $(3)$ from above
$$
\begin{aligned}
    (3)\ & \tilde{z}_{ji} = z_{ji} + \epsilon_{ij}  \text{ where } \epsilon_{ij} \overset{iid.}{\sim} N(0, \sigma^2_{\epsilon, j}), 
\end{aligned}
$$
which implies 
$$
\begin{aligned}
    \tilde{z}_{ij} \mid z_{ij} \sim N(z_{ij}, \sigma^2_{\epsilon, j})
\end{aligned}
$$
where $\sigma^2_{\epsilon, j}$ is given (Identifiability). 

**C) Exposure Model.**

Change the empirical KDE to describe $z$, not $x$ as before.
$$
\begin{aligned}
    \boldsymbol{z}_{i} \sim f_p
\end{aligned}
$$
This is easily achieved by supplying the data already in terms of $z$. 

$\rightarrow$ For implementation, all I need to change is 
1. The data: $z$ instead of $x$
2. Specify how to go from $z$ to $x$ for the linear model.

#### Unnormalized Posterior ####
Instead of expressing the posterior in $x$ bzw. $\tilde{x}$, we express it in $z$ bzw. $\tilde{z}$ and re-transform out samples to $x$ because the NUTS sampler samples from the real line and we cannot be sure that $x$ allows all real values, but we know that $z$ does. 

For this to work, we just use the empirical CDF and the inverse std. normal CDF to obtain $z$-values bzw. $\tilde{z}$ and add these to the design matrix such that we can use it as any covariate in our model. Given $\tilde{x}$ and the empirical CDF of $x$, this is reproducible for application after data sharing. 
$$
\begin{aligned}
    &p(\boldsymbol{Z}, \boldsymbol{\beta}, \sigma^2 \mid \boldsymbol{\tilde{Z}}, \boldsymbol{y}, \boldsymbol{\sigma}^2_{\epsilon})\\
   &\propto
\left( 
\prod_{i=1}^{n}{
    p(y_{i}\mid \boldsymbol{z}_{i}, \boldsymbol{\beta}, \sigma^2)
    p(\boldsymbol{\tilde{z}}_i \mid \boldsymbol{z}_i)
    p(\boldsymbol{z}_i)
}
\right)
p(\boldsymbol{\beta})
p(\sigma^2)
\\

&= 
\left( 
\prod_{i=1}^{n}{
     \left[N(\boldsymbol{x(z_i)}^T\boldsymbol{\beta}, \sigma^2)\right]
   \left[N(\boldsymbol{z}_i, \sigma^2_{\epsilon})\right]
   \left[f_p(\boldsymbol{z}_i)\right]
}
\right)
N(\boldsymbol{0}, b^2 \boldsymbol{I}_{p \times p})
Ga(c, d)
\end{aligned}
$$

Denote the row of the design matrix in terms of $\boldsymbol{z}_i$ by $(1, \boldsymbol{x(z_i)})$: 
$$
\begin{align*}
\propto &
\left(  (\sigma^2)^{-n / 2}\prod_{i=1}^{n}{
\left[  
    \exp\left( -\frac{1}{2\sigma^2} (y_i - (1, \boldsymbol{x(z_i)})\boldsymbol{\beta})^2 \right)
\right]
\left[  
    \exp\left( -\frac{1}{2} (\boldsymbol{\tilde{z}}_i - \boldsymbol{z}_i)^T \boldsymbol{G}^{-1}(\boldsymbol{\tilde{z}}_i - \boldsymbol{z}_i ) \right)
\right] 
\left[ f_p(\boldsymbol{z}_i) \right]
}
\right)\\
& \times \det(b^2 \boldsymbol{I}_{p \times p})^{-1 / 2} \exp\left( - \frac{1}{2b^2} \boldsymbol{\beta}^T \boldsymbol{\beta} \right) (\sigma^2)^{c-1} \exp\left( -d \sigma^2 \right) \\
\end{align*}
$$


The log posterior is (up to a constant $C$) equal to 
$$
\begin{align*}
     \log & [p(\boldsymbol{z}, \boldsymbol{\beta}, \sigma^2 \mid \boldsymbol{\tilde{z}}, \boldsymbol{y}, \boldsymbol{G})]\\
    & = C + -\frac{n}{2} \log \sigma^2 \\
    & + \sum_{i=1}^{n}{
\left( 
-\frac{1}{2\sigma^2}(y_i - (1, \boldsymbol{x(z_i))}^T \boldsymbol{\beta})^2 
- \frac{1}{2} (\boldsymbol{\tilde{z}}_i - \boldsymbol{z}_i)^T \boldsymbol{G}^{-1}(\boldsymbol{\tilde{z}}_i - \boldsymbol{z}_i) 
+ \log f_p(\boldsymbol{z}_i)
\right)} \\
    & - \frac{p}{2} \log b^2 - \frac{1}{2b^2} \boldsymbol{\beta}^T \boldsymbol{\beta} \\
    & + (c-1) \log \sigma^2 - d \sigma^2 \\
\end{align*}
$$
Apply change of variable to $\upsilon := \log \sigma^2$: 
$$
\begin{align*}
     \log & [p(\boldsymbol{x}, \boldsymbol{\beta}, \upsilon\mid \boldsymbol{\tilde{x}}, \boldsymbol{y}, \boldsymbol{G})]\\
    & = C + -\frac{n}{2} \upsilon \\
    & + \sum_{i=1}^{n}{\left( -\frac{1}{2 \exp \upsilon}(y_i - (1, \boldsymbol{x(z_i)})^T \boldsymbol{\beta})^2 - \frac{1}{2} (\boldsymbol{\tilde{z}}_i - \boldsymbol{z}_i)^T G^{-1}(\boldsymbol{\tilde{z}}_i - \boldsymbol{z}_i) + \log f_p(\boldsymbol{z}_i)\right)} \\
    & - \frac{p}{2} \log b^2 - \frac{1}{2b^2} \boldsymbol{\beta}^T \boldsymbol{\beta} \\
    & + (c-1) \upsilon - d \exp \upsilon  \underbrace{+ \upsilon}_{+ \log |J|}\\
\end{align*}
$$


#### Continuous Rank Transformation ####
For `JAX` to be happy, I need an invertible and differentiable function within the  ePIT calculation step.In other words, $\hat{F}$ needs to be continuous and invertible. I know that `JAX` has some `sigmoid` and its inverse implemented. Thus, I can fit a logit model, extract the parameters and have an invertible and continuous proxi of the eCDF (right?; While not accounting for any uncertainty. But nor do pobs - which are not a proxi. lol).

Following uses the `DR1TKCAL` column of the `voe_data` with $405$ observations. 
1. Visualize the empirical CDF: 
Use the definition of the empirical CDF to just plot it with `kcal` on the x-axis. 

2. I have one data point per obs which is basically my "empirical sigmoid". 
I just apply the inverse of the actual sigmoid to obtain the "linear predictor" to which I fit a linear regression. Then I obtain the $\beta$s to describe the sigmoid. 

3. Use sigmoid and its inverse to obtain pobs and use it in the NUTS to transform observations. 
    - Seems to work! Specified the sigmoid and inverse sigmoid functions. 



---

The following was fun, but is garbage: 

To ease the application of **change of variable**, we thought of a continuous expression for the empirical CDF. 
$$
\begin{aligned}
    p(x) &:= \hat{F}^{(cont)}(x) = \frac{1}{n+1} \left( 1 + \frac{n-1}{x_{max}-x_{min}}(x-x_{min}) \right) \in [0, 1)\\ 
    \hat{F}'^{(cont)}(x) &= \frac{\delta \hat{F}^{(cont)}}{\delta x}(x) = \frac{n-1}{(n+1)(x_{max}-x_{min})}
\end{aligned}
$$
**Derivation of functional Form**: 
- Lowest value must be assigned rank $1$. 
- Highest value must be assigned rank $n$.

$\rightarrow$ continuous proxi of this discrete, i.e. *non-continuous*, empirical function is just the slope-triangle ("Steigungsdreieck"). 
Additionally, we transform the continuous proxi into the interval of $[0, 1)$ following the logic of the common empirical CDF. 




### ePIT error
- Could I avoid sharing the true distribution of the variables $x$ and instead share the distribution of variables $z$? Then the information about the empirical density itself is contained in the transformation, but one cannot backtrack to the original values? Because that would require the actual empirical density? 










### Berkson error ###
TODO

#### Mathematical Definition ####

#### Model Specification ####

#### Unnormalized Posterior ####

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
