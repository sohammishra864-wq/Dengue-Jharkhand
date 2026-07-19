
import pandas as pd
import numpy as np
import os

MASTER = os.path.join(
    "outputs",
    "master_features.csv"
)

QC_OUT = os.path.join(
    "outputs",
    "qc"
)

os.makedirs(QC_OUT, exist_ok=True)

print("\n")
print(" POST-MERGE QC")
df = pd.read_csv(MASTER)


print(f"\nShape: {df.shape}")
print(f"Districts: {df['District'].nunique()}")
print(f"Years: {df['Year'].min()}–{df['Year'].max()}")


dups = df.duplicated(
    subset=["District", "Year", "Month"]
).sum()

print()
print("DUPLICATE CHECK")

if dups == 0:
    print(" No duplicate district-month rows")
else:
    print(f" Found {dups} duplicate rows")


print()
print("TEMPORAL CONTINUITY")

gaps = []

for district in sorted(df["District"].unique()):

    sub = df[df["District"] == district]

    ym = set(
        zip(sub["Year"], sub["Month"])
    )

    years = range(
        int(sub["Year"].min()),
        int(sub["Year"].max()) + 1
    )

    expected = set()

    for y in years:
        for m in range(1, 13):
            expected.add((y, m))

    missing = sorted(expected - ym)

    if len(missing) > 0:
        gaps.append((district, len(missing)))

if len(gaps) == 0:
    print(" No temporal gaps")
else:
    print(" Temporal gaps detected")
    for g in gaps[:10]:
        print(g)


print()
print("FEATURE MISSINGNESS")

miss_table = []

for c in df.columns:

    miss = df[c].isna().sum()
    pct = (miss / len(df)) * 100

    miss_table.append([c, miss, round(pct, 2)])

miss_df = pd.DataFrame(
    miss_table,
    columns=["Feature", "Missing", "Pct"]
)

print(
    miss_df.sort_values(
        "Pct",
        ascending=False
    ).to_string(index=False)
)

miss_df.to_csv(
    os.path.join(QC_OUT, "postmerge_missingness.csv"),
    index=False
)


print()
print("DISTRICT COVERAGE")

coverage = df.groupby("District").size()

print(coverage)


print()
print("RANGE VALIDATION")

checks = {
    "Rainfall_mm": (0, 2000),
    "NDVI": (-1, 1),
    "Humidity_pct": (0, 100),
    "LST_C": (-10, 70),
    "LST_Night_C": (-10, 70),
    "SoilMoist_m3m3": (0, 1),
}

for col, (low, high) in checks.items():

    if col not in df.columns:
        continue

    vals = df[col].dropna()

    bad = ((vals < low) | (vals > high)).sum()

    if bad == 0:
        print(f" {col}: OK")
    else:
        print(f" {col}: {bad} invalid rows")


summary_path = os.path.join(
    QC_OUT,
    "postmerge_summary.txt"
)

with open(summary_path, "w", encoding="utf-8") as f:

    f.write("POST-MERGE QC COMPLETE\n")
    f.write(f"Shape: {df.shape}\n")
    f.write(f"Duplicates: {dups}\n")
    f.write(f"Temporal gaps: {len(gaps)}\n")

print("\n")
print(" POST-MERGE QC COMPLETE")
