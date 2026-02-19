# Measurement for Privacy


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
\begin{align}
    y_{i, lbxt4} \mid \boldsymbol{\beta}, \sigma^2 &\sim N(\mu_i, \sigma^2)\\
    \mu_i &= \beta_0 + \beta_{age} x_{i, age} + \beta_{bmi} x_{i, bmi} + \beta_{kcal} x_{i, kcal}
\end{align}
$$
2. Priors
$$
\begin{align}
    \beta_j &\overset{iid.}{\sim} N(0, b^2) \text{ with given }b \\
    \sigma^2 &\sim Ga(c, d) \text{ with given }c, d
\end{align}
$$

Posterior: 
$$
\begin{align}
    \log(p(\beta, \sigma^2\mid \boldsymbol{y})) &\propto -\frac{n}{2}log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}{(y_i - \boldsymbol{x}_i^T \boldsymbol{\beta})^2} \\
    &- \frac{1}{2b^2} \sum_{j=1}^{p}{\beta_{j}^2} + (c-1) \log(\sigma^2) - d \sigma^2 + C
\end{align} 
$$

### Naive Model ###
Model to quantify the effect of the measurement error correction: Same as above, but use the erroneous observations:
$$
\begin{align}
    y_{i, lbxt4} &\sim N(\mu_i, \sigma^2)\\
    \mu_i &= \beta_0 + \beta_{age} \tilde{x}_{i, age} + \beta_{bmi} \tilde{x}_{i, bmi} + \beta_{kcal} \tilde{x}_{i, kcal}
\end{align}
$$
Rest as above, but with $\tilde{x}_{ij}$.

### ME correction ###
We correct for the measurement error using latent variables and separation into different sub-models. 
#### A) Linear Model ####
Simple linear model (Connects exposure und covariates, $p(y|x)$): 
$$
\begin{align}
    y_{lbxt4} &\sim N(\mu, \sigma^2)\\
    \mu &= \beta_0+ \beta_{age} x_{age} + \beta_{bmi} x_{bmi} + \beta_{kcal} x_{kcal} \\

\end{align}
$$
Priors
$$
\begin{align}
    \beta_j &\overset{iid.}{\sim} N(0, b^2) \text{ with given }b \\
    \sigma^2 &\sim Ga(c, d) \text{ with given }c, d
\end{align}
$$
Same all linear models above. Note that this model uses the *true* values of the covariates $x_{ij}$. 
#### B) Measurement Model ####
This model specifies how the measurement error is applied to the signal. 

e.g.: Gaussian, additive error:
$$
\begin{align}
    \tilde{x}_{ij} &= x_{ij} + \epsilon_{ij} \text{ with } \epsilon \sim N(0, \sigma^2_{\epsilon})\\
    \sigma^2_{\epsilon} &\sim Ga(e, f) \text{ with given }e, f
\end{align}
$$

#### C) Signal Model ####
Finally, the signal models specifies the distribution of the signal itself, i.e. distributional assumptions on latent (true) variable $x$. 
$$
\begin{align}
    x_{ij} \sim F_p
\end{align}
$$

**Important.** Assume a reasonable distribution $F_p$ here. Else, we might run into a bias. For now, we can use the empirical density of the original variable (where "original" refers to the variable without error); 
This might be too much information published, but maybe not? 
    - Can supply empirical distribution as function to JAX (I think). 
    - How problematic is a publication of the error-free data distribution? 

#### Likelihood ####
$$
\begin{align}
    p(\boldsymbol{\beta}, x | y, \tilde{x}) \propto p(\sigma^2)p(\sigma^2_{\epsilon})p(\boldsymbol{\beta})\prod_{i=1}^{n}{p(y_i|\boldsymbol{\beta}, x_i) p(\tilde{x_i}|x_i, \sigma_\epsilon)}p(x_i)
\end{align}
$$

Für priors: 
1. Normalverteilung mit hoher Varianz 
2. Simulation 

! Prior predictive checks


## Errors
In the following, we assume a total of $K$ variables where $j \in \{ 1, ..., K\}$ denotes the $j$-th variable. Each variable is observed a $n$ times with $i = 1, ..., n$ denoting the $i$-th observation.

*Note: An error should only increase the variance and not introduce any bias!* 

#### Evaluation of the Error-Degree ####
This section presents the errors we added to the data. 
The uncertainty introduced by each error is evaluated using the uncertainty evaluation formula (UEF) defined as 
$$
\begin{align}
    nMSE = \frac{(x_{true} - x_{error})^2}{\text{Var}(x_{true})}
\end{align}
$$

We aim to have each error introduce the same level of uncertainty such that results are comparable among error types. 

### Additive normal error 
To each variable, we just add a normally distributed random variable with an expected value of $0$. 
The variance of the normal distribution affects the UEF. 

#### Mathematical definition ####
$$
\begin{align}
\begin{split}
    \tilde{x}_{ji} = x_{ji} + \epsilon_{ji} \\
    \text{where } \epsilon_{ji} \overset{indep.}{\sim} N(0, \sigma^2_{\epsilon, j}) 
\end{split}
\end{align}
$$
**Note:** Allow for variable $j$ specific error variance $\sigma^{2}_{\epsilon, j}$!!, i.e. within a variable, the error variance is constant, but among variables, the error variance may differ
(Thus, not iid., but only independently distributed).
#### Model Specification ####
**A) Linear Model**. 

Likelihood. 
$$
\begin{align}
    y_{i, lbxt4} &\sim N(\mu_i, \sigma^2)\\
    \mu_i &= \beta_0 + \beta_{age} x_{i, age} + \beta_{bmi} x_{i, bmi} + \beta_{kcal} x_{i, kcal} \\
\end{align}
$$
Priors. 
$$
\begin{align}
    \boldsymbol{\beta} &\sim  N(\boldsymbol{0}, b^2 \boldsymbol{I}_{p \times p}) \text{ with given }b, \\
    \sigma^2 &\sim Ga(c, d) \text{ with given }c, d. 
\end{align}
$$
**B) Measurement Model.**
$$
\begin{align}
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
\end{align}
$$
Priors. 
$$
\begin{align}
    \sigma^2_{\epsilon, j} \sim Ga(e, f) \text{ with given }e, f. 
\end{align}
$$

**C) Signal Model.**
$$
\begin{align}
    \boldsymbol{x}_{i} \sim f_p
\end{align}
$$
where $f_p$ is the joint empirical density of $X_{age}, X_{bmi}, X_{kcal}$ which is evaluated at $x_{i, age}, x_{i, bmi}, x_{i, kcal}$ in the unnormalized posterior likelihood. In `JAX`, this is achieved by passing a callable function to the log-density. 

#### Unnormalized Posterior ####
$$
\begin{align}
    &p(x, \boldsymbol{\beta}, \sigma^2, \sigma^2_{\epsilon, age}, \sigma^2_{\epsilon, bmi}, \sigma^2_{\epsilon, kcal}\mid \boldsymbol{\tilde{x}}, \boldsymbol{y})\\
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
\underbrace{\left( \prod_{j \in \{age, bmi, kcal\}}^{ }{p(\sigma^2_{\epsilon, j})} \right)
}_{p(\boldsymbol{\sigma}^2_{\epsilon})}\\

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
\left( \prod_{j \in \{age, bmi, kcal\}}^{ }{Ga(e, f)} \right)\\
\end{align}
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
    \det(\boldsymbol{G})^{-1 / 2} \exp\left( -\frac{1}{2} (\boldsymbol{\tilde{x}}_i - \boldsymbol{x}_i   )^T \boldsymbol{G}^{-1}(\boldsymbol{\tilde{x}}_i - \boldsymbol{x}_i ) \right)
\right] 
\left[ f_p(\boldsymbol{x}_i) \right]
}
\right)\\
& \times \det(b^2 \boldsymbol{I}_{p \times p})^{-1 / 2} \exp\left( - \frac{1}{2b^2} \boldsymbol{\beta}^T \boldsymbol{\beta} \right) (\sigma^2)^{c-1} \exp\left( -d \sigma^2 \right) \\
& \times \prod_{j \in \{age, bmi, kcal\}}^{ }{(\sigma^2_{\epsilon, j})^{e-1}\exp\left( -f \sigma^2_{\epsilon, j} \right)}
\end{align*}
$$

The log posterior in the given by 
$$
\begin{align*}
     \log & [p(\boldsymbol{x}, \boldsymbol{\beta}, \sigma^2 , \boldsymbol{\sigma}^2_{\epsilon}\mid \boldsymbol{\tilde{x}}, \boldsymbol{y})]\\
    & \propto -\frac{n}{2} \log \sigma^2 - \frac{n}{2} \left( \log \sigma^2_{\epsilon, age} + \log\sigma^2_{\epsilon, bmi}+\log \sigma^2_{\epsilon, kcal} \right)\\
    & + \sum_{i=1}^{n}{\left( -\frac{1}{2\sigma^2}(y_i - \boldsymbol{z}_i^T \boldsymbol{\beta})^2 - \frac{1}{2} (\boldsymbol{\tilde{x}}_i - \boldsymbol{x}_i)^T \boldsymbol{G}^{-1}(\boldsymbol{\tilde{x}}_i - \boldsymbol{x}_i) + \log f_p(\boldsymbol{x}_i)\right)} \\
    & - \frac{p}{2} \log b^2 - \frac{1}{2b^2} \boldsymbol{\beta}^T \boldsymbol{\beta} \\
    & + (c-1) \log \sigma^2 - d \sigma^2 \\
    & + (e-1) (\log\sigma^2_{\epsilon, age} + \log\sigma^2_{\epsilon, bmi} + \log\sigma^2_{\epsilon, kcal}) - f (\sigma^2_{\epsilon, age} + \sigma^2_{\epsilon, bmi} + \sigma^2_{\epsilon, kcal})
\end{align*}
$$
To sample from real line, express the posterior in terms of $\log \sigma^2 =: \upsilon$ bzw. $\log \sigma^2_{\epsilon, j} =: \upsilon_{\epsilon, j}$.  
Use change of variable and simply add the $\log |J|$ for each transformation (Note: Only relevant for the densities related to the variances, but express full density in terms of log variance). 
$$
\begin{align*}
     \log & [p(\boldsymbol{x}, \boldsymbol{\beta}, \upsilon , \boldsymbol{\upsilon}_{\epsilon}\mid \boldsymbol{\tilde{x}}, \boldsymbol{y})]\\
    & \propto -\frac{n}{2} \upsilon - \frac{n}{2} \left( \upsilon_{\epsilon, age} + \upsilon_{\epsilon, bmi}+\upsilon_{\epsilon, kcal} \right)\\
    & + \sum_{i=1}^{n}{\left( -\frac{1}{2 \exp \upsilon}(y_i - \boldsymbol{z}_i^T \boldsymbol{\beta})^2 - \frac{1}{2} (\boldsymbol{\tilde{x}}_i - \boldsymbol{x}_i)^T \Upsilon_{\epsilon}^{-1}(\boldsymbol{\tilde{x}}_i - \boldsymbol{x}_i) + \log f_p(\boldsymbol{x}_i)\right)} \\
    & - \frac{p}{2} \log b^2 - \frac{1}{2b^2} \boldsymbol{\beta}^T \boldsymbol{\beta} \\
    & + (c-1) \upsilon - d \exp \upsilon  \underbrace{+ \upsilon}_{+ \log |J|}\\
    & + (e-1) (\upsilon_{\epsilon, age} + \upsilon_{\epsilon, bmi} + \upsilon_{\epsilon, kcal}) - f (\exp \upsilon_{\epsilon, age} + \exp \upsilon_{\epsilon, bmi} + \exp \upsilon_{\epsilon, kcal}) \\
    & \underbrace{+ \upsilon_{\epsilon, age} + \upsilon_{\epsilon, bmi} + \upsilon_{\epsilon, kcal}}_{+ \log |J_{\epsilon, age}| + \log |J_{\epsilon, bmi}| + \log |J_{\epsilon, kcal}|}
\end{align*}
$$
where $\Upsilon_{\epsilon} = diag(\exp \upsilon_{\epsilon, age}, \exp \upsilon_{\epsilon, bmi}, \exp \upsilon_{\epsilon, kcal})$. 

### Multiplicate log-normal error
Every variable is multiplied by a log-normally distributed random variable. 

#### Mathematical definition
$$
\begin{align}
\begin{split}
    \tilde{x}_{ji} &= x_{ji} \cdot \epsilon_{ji} \\
    &\text{where } \log(\epsilon_{ji}) \overset{iid.}{\sim}N\left(\mu_{(log)} , \sigma_{(log)}^2\right) \\ 
    &\leftrightarrow \epsilon_{ij} \overset{iid.}{\sim} \text{Lognormal}\left(\exp\left( \mu_{(log)}+ \frac{\sigma_{(log)}^2}{2} \right) , \left[\exp(\sigma^2)-1\right]\exp\left( 2 \mu_{(log)}+\sigma_{(log)} \right) \right).
\end{split}
\end{align}
$$
To have $\mathbb{E}_{ }\left[ \tilde{x}_{ij}\right] = x_{ij}$, choose $\mathbb{E}_{}\left[ \epsilon_{ij}\right] = 1$, i.e. $\mu_{(log)}= -\frac{\sigma^2_{(log)}}{2}$. 
### ePIT error
For this error, we make use of the empirical probability integral transform (ePIT) bzw. the empirical CDF function. @okhrinBasicElemts2017, p. 195 defines this as 
$$
\begin{align}
    \hat{F}_{j}(x) = \frac{1}{n+1} \sum_{k=1}^{n}{I(x_{jk} \leq x)}
\end{align}
$$
For each $x_{ji} \in \{x_{j1}, ..., x_{jn}\}$, we define $p_{ji} := \hat{F}_{j}(x_{ji})$. Due to the properties of the (empirical) CDF, we know that $p_{ji} \overset{iid.}{\sim} \text{Uniform}(0, 1)$. Note that the empirical CDF basically assigns normalized ranks. Thus, the following uses the term "rank" somewhat interchangeable with the empirical CDF value as it is pretty much the same. 

*Note: $\hat{F}_{j}$ is **not** continuous! For every $x \in (x_{j(k)}, x_{j(k+1)})$, it is constant ($x_{j(k)}$ denotes the $k$-th ordered observation.)*

From here, we can add any type of error from any distribution by 
1) transforming $p_{ij}$ using a quantile function of any distribution which would yield $Q_{}(p_{ij})$ to then follow this distribution where $Q$ denotes the quantile function of this chosen distribution
2) We add an error onto $Q(p_{ij})$ of any kind, i.e. of any distribution
3) We retransform $Q(p_{ij})$ using the *known* CDF bzw. inverse of $Q$ to again obtain a noisy $p_{ij}$. 
4) From noisy $p_{ij}$ we can then go back to the original scale using $\hat{F}^{-1}_{j}$ from the above equation.

Of course, we could skip steps if we would just add noise on the uniform variable $p_{ij}$. Maybe this would be more efficient and / or easier to model? 

For now, we used the std. normal quantile function, added a std. normal error and then re-transformed the variable to its original scale using the empirical CDF. 

Overall, this approach has the upside of never introducing new values. Instead, all noisy data points are still real data points. 
Essentially: </br>
This type of error build rank-value pairs. Then takes every observation and potentially (that is, most likely) assigns it a new rank. 
When all observation has been assigned a new rank, each rank is the value assigned which was saved in the rank-value pair. 
Rephrasing it like this seems to simplify things: It basically is an error on the assigned rank. 





#### Mathematical Error Definition
$$
\begin{align}
\begin{split}
    (1)\ & p_{ji} = \hat{F}_{j}(x_{ji})\\
    (2)\ & z_{ji} = \Phi^{-1}(p_{ji}) \\
    (3)\ & \tilde{z}_{ji} = z_{ji} + \epsilon_{i}  \text{ where } \epsilon_{i} \overset{iid.}{\sim} N(0, 1)\\
    (4)\ & \tilde{p}_{ji} = \Phi(\tilde{z}_{ji}) \\
    (5) \ & \tilde{x}_{ji} = \hat{F}^{-1}_{j}(\tilde{p}_{ji})
\end{split}
\end{align}
$$

In short: 
$$
\begin{align}
    \tilde{x}_{ji} = \hat{F}^{-1}_{j}(
        \Phi\{ 
            \Phi^{-1}[
                \hat{F}_{j}(x_{ji})
            ] + \epsilon_{i}
       \} 
    )
\end{align}
$$

This expression can be generalized, I think: 
$$
\begin{align}
    \tilde{x}_{ji} = \hat{F}^{-1}_{j}(
        G(
                \hat{F}_{j}(x_{ji})
        )
    )
\end{align}
$$
where $G(.)$ is just any function that (potentially) assigns a new rank to $x_{ji}$ and $\hat{F}$ (re-)transform from rank to (observed) value. 
In the above case, 




$$
\begin{align}
    G(y) = \Phi(\Phi^{-1}(y) + \epsilon) \text{ where } \epsilon \overset{iid.}{\sim}N(0, 1)
\end{align}
$$

<!-- Also ich glaube, es funktioniert nicht so schön, wie du das aufgeschrieben hast -->

*Note: I expect that, depending on the added error, the distribution of $\tilde{x}_{ji}$ flattens compared to the original distribution. But because values cannot exceed the largest observed value, I expect a peaky behaviour towards the edges. Let's see!* :)

#### Posterior ####
Instead of expressing the posterior in $x$ bzw. $\tilde{x}$, we express it in $z$ bzw. $\tilde{z}$ and re-transform out samples to $x$ because the NUTS sampler samples from the real line and we cannot be sure that $x$ allows all real values, but we know that $z$ does. 

For this to work, we just use the empirical CDF and the inverse std. normal CDF to obtain $z$-values bzw. $\tilde{z}$ and add these to the design matrix such that we can use it as any covariate in our model. Given $\tilde{x}$ and the empirical CDF of $x$, this is reproducible for application after data sharing. 
$$
\begin{align}
    p(\boldsymbol{\beta}, z | y, \tilde{z}) \propto p(\boldsymbol{\beta})\prod_{i=1}^{n}{p(y_i|\boldsymbol{\beta}, z_i) p(\tilde{z_i}|z_i, \sigma_\epsilon)}p(z_i)
\end{align}
$$


#### Continuous Rank Transformation ####
To ease the application of **change of variable**, we thought of a continuous expression for the empirical CDF. 
$$
\begin{align}
    p(x) &:= \hat{F}^{(cont)}(x) = \frac{1}{n+1} \left( 1 + \frac{n-1}{x_{max}-x_{min}}(x-x_{min}) \right) \in [0, 1)\\ 
    \hat{F}'^{(cont)}(x) &= \frac{\delta \hat{F}^{(cont)}}{\delta x}(x) = \frac{n-1}{(n+1)(x_{max}-x_{min})}
\end{align}
$$
**Derivation of functional Form**: 
- Lowest value must be assigned rank $1$. 
- Highest value must be assigned rank $n$.

$\rightarrow$ continuous proxi of this discrete, i.e. *non-continuous*, empirical function is just the slope-triangle ("Steigungsdreieck"). 
Additionally, we transform the continuous proxi into the interval of $[0, 1)$ following the logic of the common empirical CDF. 

### Berkson error ###
TODO


# TODO #
- Quantify effect of distributional-misspecification (e.g. of the distribution of the real data). 
