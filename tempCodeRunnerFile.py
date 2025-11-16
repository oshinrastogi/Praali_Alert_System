import pandas as pd
import glob

# Step 1: Read all CSV files inside dataFIles folder
files = glob.glob("dataFIles/*.csv")

dfs = []

for file in files:
    df = pd.read_csv(file)
    
    # Step 2: Move 'labelled_prali' column to the END
    if "prali_fire" in df.columns:
        last_col = df.pop("prali_fire")
        df["prali_fire"] = last_col
    
    dfs.append(df)

# Step 3: Merge all rows
merged_df = pd.concat(dfs, ignore_index=True)

# Step 4: Save output as final_data.csv
merged_df.to_csv("merged_data.csv", index=False)

print("Done! Merged file saved as final_data.csv")

