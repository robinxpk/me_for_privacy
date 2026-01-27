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

## Errors

This section presents the errors we added to the data. The uncertainty introduced by each error is evaluated using the uncertainty evaluation formula (UEF) defined as 
$$
\begin{align}
    \frac{1}{2}.
\end{align}
$$
We aim to have each error introduce the same level of uncertainty such that results are comparable among error types. 
The following briefly introduces each error and gives a mathematical definition. 

In the following, we assume a total of $K$ variables where $j \in \{ 1, ..., K\}$ denotes the $j$-th variable. Each variable is observed a $n$ times with $i = 1, ..., n$ denoting the $i$-th observation.

*Note: An error should only increase the variance and not introduce any bias!* 

### Additive normal error 
To each variable, we just add a normally distributed random variable with an expected value of $0$. 
The variance of the normal distribution affects the UEF. 

#### Mathematical definition
$$
\begin{align}
\begin{split}
    \tilde{x}_{ji} = x_{ji} + \epsilon_{i} \\
    \text{where } \epsilon_{j} \overset{iid.}{\sim} N(0, \sigma^{(add.)}_{\epsilon}) 
\end{split}
\end{align}
$$

<!-- Is the notation correct? I assume we choose a different error for each observation so that would result in -->

$$
\begin{align}
\begin{split}
    \tilde{x}_{ji} = x_{ji} + \epsilon_{ji} \\
    \text{where } \epsilon_{ji} \overset{iid.}{\sim} N(0, \sigma^{(add.)}_{\epsilon, j}) 
\end{split}
\end{align}
$$

### Multiplicate log-normal error
Every variable is multiplied by a log-normally distributed random variable. 

#### Mathematical definition
$$
\begin{align}
\begin{split}
    \tilde{x}_{j} = x_{j} \cdot \epsilon_{j} \\
    \text{where } \log(\epsilon_{j}) \overset{iid.}{\sim}N(0, \sigma^{(mult.)}_{\epsilon})
\end{split}
\end{align}
$$

<!-- same problem right? -->
$$
\begin{align}
\begin{split}
    \tilde{x}_{ji} = x_{ji} \cdot \epsilon_{ji} \\
    \text{where } \log(\epsilon_{ji}) \overset{iid.}{\sim}N(0, \sigma^{(mult.)}_{\epsilon, j})
\end{split}
\end{align}
$$

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

#### Mathematical definition
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
<!-- brutal lol -->
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
<!-- $$
\begin{align}
    G(y) = \Phi(\Phi^{-1}(y + \epsilon)) \text{ where } \epsilon \overset{iid.}{\sim}N(0, 1)
\end{align}
$$ -->

<!-- I think the equation is / was wrong because $\Phi^{-1} (x + \epsilon) \neq \Phi^-1(x) + \epsilon -->
$$
\begin{align}
    G(y) = \Phi(\Phi^{-1}(y) + \epsilon) \text{ where } \epsilon \overset{iid.}{\sim}N(0, 1)
\end{align}
$$

<!-- Also ich glaube, es funktioniert nicht so schön, wie du das aufgeschrieben hast -->

*Note: I expect that, depending on the added error, the distribution of $\tilde{x}_{ji}$ flattens compared to the original distribution. But because values cannot exceed the largest observed value, I expect a peaky behaviour towards the edges. Let's see!* :)


### Berkson error ###








