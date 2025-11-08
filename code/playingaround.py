import pandas as pd

data_path = r"data/"

nhanes_demo = pd.read_sas(f"{data_path}DEMO_L.xpt")
nhanes_diet1 = pd.read_sas(f"{data_path}DR1IFF_L.xpt")

nhanes = pd.concat([nhanes_demo, nhanes_diet1], ignore_index = True, sort = False)

test = nhanes_demo.merge(
    nhanes_diet1,
    on="SEQN",
    how="left",     # keep everyone in demo
    validate="1:1"  # optional, see below
)

