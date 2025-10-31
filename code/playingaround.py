# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
data_path = r"data/"

nhanes_demo = pd.read_sas(f"{data_path}DEMO_L.xpt")
nhanes_diet1 = pd.read_sas(f"{data_path}DR1IFF_L.xpt")


key = "SEQN"

# make sure key has the same dtype
nhanes_demo[key] = nhanes_demo[key].astype("int64")
nhanes_diet1[key] = nhanes_diet1[key].astype("int64")

# join side-by-side (each food item gets the person’s demographics)
nhanes = pd.merge(
    nhanes_demo, nhanes_diet1, on=key, how="inner", validate="one_to_many"
)

# nhanes = pd.concat([nhanes_demo, nhanes_diet1], ignore_index=True, sort=False)


# %%
# Set up automatic layout
num_cols = len(nhanes.columns)
ncols = 3  # number of plots per row
nrows = (num_cols + ncols - 1) // ncols  # rows needed

plt.figure(figsize=(15, 4 * nrows))

for i, col in enumerate(nhanes.columns, 1):
    plt.subplot(nrows, ncols, i)

    if pd.api.types.is_numeric_dtype(nhanes[col]):
        nhanes[col].dropna().plot(kind="hist", bins=30, edgecolor="black")
        plt.title(f"{col} (Histogram)")
    else:
        nhanes[col].value_counts(dropna=False).plot(kind="bar")
        plt.title(f"{col} (Bar Plot)")

    plt.tight_layout()

plt.show()


# %%

nhanes_2 = nhanes.loc[:, ["RIDAGEYR", "DR1IKCAL"]]
# %%
