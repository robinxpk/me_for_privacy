import pandas as pd

# %%
nhanes_data_path = r"data/DEMO_L.xpt"

nhanes = pd.read_sas(nhanes_data_path)


print("Done")

# %
# nhanes_rename_dict = {
#     # Format: {"newname": "oldname", ...}
#     "SEQN": "id"
# }
# nhanes.rename(
#     index = nhanes_rename_dict
# )

# %%
