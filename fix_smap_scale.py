import pandas as pd
import os

path = os.path.join(
    "data",
    "raw",
    "jhk_smap_2015_2023.csv"
)

df = pd.read_csv(path)

print("\nBEFORE SCALING ")
print(df["mean"].describe())

df["mean"] = df["mean"] / 100

print("\nAFTER SCALING ")
print(df["mean"].describe())

df.to_csv(path, index=False)

print("\n SMAP scaling corrected and saved")