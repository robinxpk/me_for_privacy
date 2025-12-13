# Measurement for Privacy

## TODO
_Alle_ Variablen Errorterm - unabhängig von socially desired value - sonst kein Schutz gegen identifizierbar

0) DBScan result for a visible number of clusters prior to error 
1) Boolean pro variable ziehen (50%?)
2) If TRUE: Fehler 
    Hierzu: 
    - Numerisch: ePIT auf Variable und Variable in Normalverteilung transformieren
        Normalen Fehlerterm aufaddieren: 
        Fehlervarianz randomly ziehen
        Grund für random draw: Kein gesondertes Behandeln der Outlier
            sonder es kann durch eine sehr hohe Fehlervarianz auch zu Outliern kommen. So erhalten Outlier nicht nur möglicherweise einen Fehlerterm, sondern bestehende Outlier werden durch mögliche neue Outlier "gemasked". 
            Natürlich kann es auch dazu kommen, dass Outlier noch weiter in die Outlier-Richtung gezogen werden. Hier machen wir uns aber erstmal keine Sorgen, weil das durch gutes Masking (hoffentlich) weniger relevant ist.
    - Kategorisch: Zufälliges Ziehen aus...
        ... möglichen Kategorien (Uniform) --> Enthält keine Informationen bzgl der empirischen Verteilung der Kategorien
        ... empirische Verteilung der Kategorien --> Möglicherweise kann so die empirische Verteilung beibehalten werden, allerdings weiß ich nicht, ob das das Signal nicht weird macht? Für uniform Fehler scheint es mir einfacher zu korrigieren
        Nicht sicher was davon besser wäre?
3) Re-Fit DBScan and display same clusters

measure for LDP? 
Plots: Selektierte Cluster - vorher u. nachher

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

Also, for Cox-models fitted during brainstorming, we used the data from [this](https://doi.org/10.1093/ije/dyaa164) paper which can be found [here](https://github.com/chiragjp/voe) (file: `nhanes9904_VoE.Rdata`)

Dietary1 dataset: [here](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Dietary&Cycle=2021-2023)

Body Measures: [here](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=2021-2023)

### The third International Stroke Trial [IST-3]
[IST-3](https://datashare.ed.ac.uk/handle/10283/1931)


### [[UMAP]]-clustering ###

Check [this](https://umap-learn.readthedocs.io/en/latest/) out. 


