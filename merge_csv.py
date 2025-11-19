import pandas as pd
import glob

files = glob.glob("dataFIles/*.csv")

dfs = []

for file in files:
    df = pd.read_csv(file)
    if "prali_fire" in df.columns:
        last_col = df.pop("prali_fire")
        df["prali_fire"] = last_col
    
    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)

merged_df.to_csv("merged_data.csv", index=False)

print("Done! Merged file saved as final_data.csv")

