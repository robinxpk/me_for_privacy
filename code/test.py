# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from ME.Data import Data

# %%
data_path = r"../data/"

nhanes_demo = pd.read_sas(f"{data_path}DEMO_L.xpt")
nhanes_diet1 = pd.read_sas(f"{data_path}DR1IFF_L.xpt")

key = "SEQN"

# make sure key has the same dtype
nhanes_demo[key] = nhanes_demo[key].astype("int64")
nhanes_diet1[key] = nhanes_diet1[key].astype("int64")

# join side-by-side (each food item gets the person’s demographics)
nhanes_raw = pd.merge(
    nhanes_demo, nhanes_diet1, on=key, how="inner", validate="one_to_many"
)


# %%

nhanes_raw = nhanes_raw.loc[:, [
        "DR1DAY", # Intake day of the week
        "DR1LANG", # Language respondent used mostly
        "RIDAGEYR", # Age
        "DR1IPROT", # Protein (gm)
        "DR1IKCAL", # Energy (kcal)
    ]
]
nhanes = nhanes_raw.dropna()


# 1) Boolean pro variable ziehen (50%?)
# 2) If TRUE: Fehler 
#     Hierzu: 
#     - Numerisch: ePIT auf Variable und Variable in Normalverteilung transformieren
#         Normalen Fehlerterm aufaddieren: 
#         Fehlervarianz randomly ziehen
#         Grund für random draw: Kein gesondertes Behandeln der Outlier
#             sonder es kann durch eine sehr hohe Fehlervarianz auch zu Outliern kommen. So erhalten Outlier nicht nur möglicherweise einen Fehlerterm, sondern bestehende Outlier werden durch mögliche neue Outlier "gemasked". 
#             Natürlich kann es auch dazu kommen, dass Outlier noch weiter in die Outlier-Richtung gezogen werden. Hier machen wir uns aber erstmal keine Sorgen, weil das durch gutes Masking (hoffentlich) weniger relevant ist.
#     - Kategorisch: Zufälliges Ziehen aus...
#         ... möglichen Kategorien (Uniform) --> Enthält keine Informationen bzgl der empirischen Verteilung der Kategorien
#         ... empirische Verteilung der Kategorien --> Möglicherweise kann so die empirische Verteilung beibehalten werden, allerdings weiß ich nicht, ob das das Signal nicht weird macht? Für uniform Fehler scheint es mir einfacher zu korrigieren
#         Nicht sicher was davon besser wäre?
# 3) Re-Fit DBScan and display same clusters

# %%
nhanes = Data(raw_data = nhanes_raw, prob = 0.5)

# %%
