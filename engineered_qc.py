import pandas as pd
import numpy as np
import os

INPUT = os.path.join(
    "outputs",
    "master_features_engineered.csv"
)

print("\n")
print(" ENGINEERED FEATURE QC")
df = pd.read_csv(INPUT)

print(f"\nShape: {df.shape}")

dups = df.duplicated(
    subset=["District", "Year", "Month"]
).sum()

print(f"\nDuplicate district-month rows: {dups}")

print("\nLag feature validation:")

lag_cols = [
    c for c in df.columns
    if "_lag" in c
]

for c in lag_cols:

    miss = df[c].isna().sum()

    print(f"{c:<35} missing={miss}")

print("\nRolling feature validation:")

roll_cols = [
    c for c in df.columns
    if "_roll" in c
]

for c in roll_cols:

    miss = df[c].isna().sum()

    print(f"{c:<35} missing={miss}")

print("\nSeasonal feature validation:")

season_cols = [
    "month_sin",
    "month_cos",
    "monsoon_flag",
    "summer_flag"
]

for c in season_cols:

    print(
        f"{c:<20} "
        f"min={df[c].min():.2f} "
        f"max={df[c].max():.2f}"
    )

print("\nAnomaly feature validation:")

z_cols = [
    c for c in df.columns
    if "_zscore" in c
]

for c in z_cols:

    print(
        f"{c:<30} "
        f"mean={df[c].mean():.2f} "
        f"std={df[c].std():.2f}"
    )

summary_path = os.path.join(
    "outputs",
    "engineered_qc_summary.txt"
)

with open(summary_path, "w", encoding="utf-8") as f:

    f.write("ENGINEERED QC COMPLETE\n")
    f.write(f"Shape: {df.shape}\n")
    f.write(f"Duplicates: {dups}\n")
    f.write(f"Columns: {len(df.columns)}\n")

print("\n")
print(" ENGINEERED QC COMPLETE")
