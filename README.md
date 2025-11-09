# Measurement for Privacy

## TODO
Alle Variablen Errorterm - unabhängig von socially desired value - sonst kein Schutz gegen identifizierbar

Boolean pro variable ziehen (50%?)
If TRUE: Fehler 
kategorisch: 
    random draw aus allen unique Kategorien --> Gar keine Informationen über distri
    Nach Verteilung in Daten --> Enthält dann implizit Information über wahre Verteilung
numerisch: 
    Emp. Inverse Probabilitity transform into Gaussian --> Error with random variance, damit Outlier nicht mehr identifizierbar und auch random entstehen können--> re-transform (Sollte gut mit Outliern dealen)


measure for LDP? 
Plots: Selektierte Cluster - vorher u. nachher



1. Draw boolean for each variable i.e. column and assign error according to value




## Virtual Environment Setup (Conda)

Based on the given `.yaml`-file, create a conda environment using: 

`conda env create -f environment.yaml`

In case one wants to update the `.yaml` file, create a new file using: 

`conda env export > environment.yaml`

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

Dietary1 dataset: [here](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Dietary&Cycle=2021-2023)

### The third International Stroke Trial [IST-3]
[IST-3](https://datashare.ed.ac.uk/handle/10283/1931)



