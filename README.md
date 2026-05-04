#### TODOs ####
- [ ] !!! For the ePIT correction, I have the wrong sigma... I implemented it such that I sample 2 log(sigma^2) or something. The samples taken by naive model are correct, though. Something is not right :D 
- [ ] DEAL WITH: Removed negative solution for normal data. 
- [ ] Fix KDE which might introduce the lognormal bias(?) in the current results (See comment on bias due to bad posterior)

Error must not depend on the data
	-> Why then use different Sigmas for each obs

- Fit Code
- Uncertainty of the estimate before and after, i.e. how does the uncertainty Change? 
- Larger Data set 
- Error on all variables 
- ePIT only for now? 
- Do we have to condition f_p on the other covariates? 
	- Haben nicht die marginal genutzt, sondern Joint. Hier sind slides falsch bzw. meine Notation der KDE falsch
	> p(x_kcal)
	> p(x_1, …, x_p) / p(x_2, …, x_p) 
- Change sampler 



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

## Running Code ## 
Potentially, before running the code, run 
```
pip install -e .
```
from within the root directory and with activated conda environment. 
This ensures that the `ME`-package is loaded properly.



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

