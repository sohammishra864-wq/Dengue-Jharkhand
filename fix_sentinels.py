
import pandas as pd
import numpy as np
import os, shutil

RAW    = os.path.join("data", "raw")
BACKUP = os.path.join("data", "raw", "backup")
os.makedirs(BACKUP, exist_ok=True)

SENTINELS = [-999, -9999, -9998, 999999, -999.0, -9999.0]

FILES = [
    "satellite_climate_district.csv",
    "jhk_night_lst_2010_2023.csv",
    "jhk_jrc_monthly_water_2010_2023.csv",
    "jhk_jrc_static_occurrence.csv",
    "jhk_smap_2015_2023.csv",
    "jhk_nightlights_2012_2023.csv",
    "jhk_static_features.csv",
]

print("\n══ SENTINEL VALUE FIXER ══\n")

for fname in FILES:
    path = os.path.join(RAW, fname)
    if not os.path.exists(path):
        print(f"  ⚠  Not found: {fname}")
        continue

    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    total_fixed = 0
    for col in numeric_cols:
        mask = df[col].isin(SENTINELS)
        n = mask.sum()
        if n > 0:
            df.loc[mask, col] = np.nan
            total_fixed += n
            print(f"  ✓ {fname} | {col}: replaced {n} sentinel values with NaN")

    if total_fixed > 0:
        shutil.copy(path, os.path.join(BACKUP, fname))
        df.to_csv(path, index=False)
        print(f"    → Saved (original backed up to data/raw/backup/)")
    else:
        print(f"  ✓ {fname}: no sentinel values found")

print("\n  Done. Re-run premerge_qc.py to verify all sentinels fixed.")