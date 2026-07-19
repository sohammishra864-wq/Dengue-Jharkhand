
import pandas as pd
import numpy as np
import os


RAW = os.path.join("data", "raw")
OUTPUT = os.path.join("outputs")

os.makedirs(OUTPUT, exist_ok=True)


def print_missing(df, col):
    miss = df[col].isna().sum()
    pct = (miss / len(df)) * 100

    print(
        f"    Missing: {miss} rows ({pct:.1f}%)"
    )

def assert_no_row_explosion(before_rows, after_rows, step_name):
    assert before_rows == after_rows, \
        f"❌ Row explosion detected after {step_name}! " \
        f"Before={before_rows}, After={after_rows}"

def summarize_merge(df, step_name):
    print(f"\n  ✓ {step_name} complete")
    print(f"    Shape: {df.shape}")
    print(f"    Districts: {df['District'].nunique()}")


print("\n════════════════════════════════════════════════════")
print("  MERGING ENVIRONMENTAL FEATURES")
print("════════════════════════════════════════════════════")

sat_path = os.path.join(RAW, "satellite_climate_district.csv")

sat = pd.read_csv(sat_path)

sat = sat.drop_duplicates(
    subset=["District", "Year", "Month"]
)

print(f"\n✓ Base satellite dataset loaded")
print(f"  Shape: {sat.shape}")
print(f"  Districts: {sat['District'].nunique()}")
print(f"  Years: {sat['Year'].min()}–{sat['Year'].max()}")


print("\n────────────────────────────────────────────────────")
print("Merging Night LST")

night = pd.read_csv(
    os.path.join(RAW, "jhk_night_lst_2010_2023.csv")
)

night = night.rename(columns={
    "mean": "LST_Night_C",
    "year": "Year",
    "month": "Month"
})

night = night.drop_duplicates(
    subset=["District", "Year", "Month"]
)

night["LST_Night_C"] = night["LST_Night_C"].round(2)

before_rows = len(sat)

sat = sat.merge(
    night[["District", "Year", "Month", "LST_Night_C"]],
    on=["District", "Year", "Month"],
    how="left"
)

assert_no_row_explosion(
    before_rows,
    len(sat),
    "Night LST merge"
)

summarize_merge(sat, "Night LST merge")
print_missing(sat, "LST_Night_C")


print("\n────────────────────────────────────────────────────")
print("Merging JRC Monthly Water")

jrc_m = pd.read_csv(
    os.path.join(RAW, "jhk_jrc_monthly_water_2010_2023.csv")
)

jrc_m = jrc_m.rename(columns={
    "mean": "Water_pct_monthly",
    "year": "Year",
    "month": "Month"
})

jrc_m = jrc_m.drop_duplicates(
    subset=["District", "Year", "Month"]
)

jrc_m["Water_pct_monthly"] = (
    jrc_m["Water_pct_monthly"].round(3)
)

before_rows = len(sat)

sat = sat.merge(
    jrc_m[
        ["District", "Year", "Month", "Water_pct_monthly"]
    ],
    on=["District", "Year", "Month"],
    how="left"
)

assert_no_row_explosion(
    before_rows,
    len(sat),
    "JRC Monthly Water merge"
)

summarize_merge(sat, "JRC Monthly Water merge")
print_missing(sat, "Water_pct_monthly")


print("\n────────────────────────────────────────────────────")
print("Merging JRC Static Water")

jrc_s = pd.read_csv(
    os.path.join(RAW, "jhk_jrc_static_occurrence.csv")
)

jrc_s = jrc_s.drop_duplicates(
    subset=["District"]
)

before_rows = len(sat)

sat = sat.merge(
    jrc_s[
        ["District", "Perm_Water_pct", "Seas_Water_pct"]
    ],
    on="District",
    how="left"
)

assert_no_row_explosion(
    before_rows,
    len(sat),
    "JRC Static Water merge"
)

summarize_merge(sat, "JRC Static Water merge")


print("\n────────────────────────────────────────────────────")
print("Merging SMAP Soil Moisture")

smap = pd.read_csv(
    os.path.join(RAW, "jhk_smap_2015_2023.csv")
)

smap = smap.rename(columns={
    "mean": "SoilMoist_m3m3",
    "year": "Year",
    "month": "Month"
})

smap = smap.drop_duplicates(
    subset=["District", "Year", "Month"]
)

smap["SoilMoist_m3m3"] = (
    smap["SoilMoist_m3m3"].round(4)
)

before_rows = len(sat)

sat = sat.merge(
    smap[
        ["District", "Year", "Month", "SoilMoist_m3m3"]
    ],
    on=["District", "Year", "Month"],
    how="left"
)

assert_no_row_explosion(
    before_rows,
    len(sat),
    "SMAP merge"
)

summarize_merge(sat, "SMAP merge")
print_missing(sat, "SoilMoist_m3m3")


print("\n────────────────────────────────────────────────────")
print("Merging Nightlights")

nl = pd.read_csv(
    os.path.join(RAW, "jhk_nightlights_2012_2023.csv")
)

nl = nl.rename(columns={
    "mean": "Nightlights",
    "year": "Year"
})

nl = nl.drop_duplicates(
    subset=["District", "Year"]
)

nl["Nightlights"] = nl["Nightlights"].round(3)

before_rows = len(sat)

sat = sat.merge(
    nl[
        ["District", "Year", "Nightlights"]
    ],
    on=["District", "Year"],
    how="left"
)

assert_no_row_explosion(
    before_rows,
    len(sat),
    "Nightlights merge"
)

summarize_merge(sat, "Nightlights merge")
print_missing(sat, "Nightlights")


print("\n────────────────────────────────────────────────────")
print("Merging Static Features")

static = pd.read_csv(
    os.path.join(RAW, "jhk_static_features.csv")
)

static = static.drop_duplicates(
    subset=["District"]
)

static_cols = [
    "District",
    "Elevation_m",
    "Slope_deg",
    "Builtup_pct",
    "Forest_pct",
    "Cropland_pct",
    "Pop_density"
]

available_static_cols = [
    c for c in static_cols if c in static.columns
]

before_rows = len(sat)

sat = sat.merge(
    static[available_static_cols],
    on="District",
    how="left"
)

assert_no_row_explosion(
    before_rows,
    len(sat),
    "Static Features merge"
)

summarize_merge(sat, "Static Features merge")


print("\n────────────────────────────────────────────────────")
print("Checking Hospital Count")

hosp_path = os.path.join(
    RAW,
    "jhk_hospital_count.csv"
)

if os.path.exists(hosp_path):

    hosp = pd.read_csv(hosp_path)

    hosp = hosp.drop_duplicates(
        subset=["District"]
    )

    before_rows = len(sat)

    sat = sat.merge(
        hosp[
            ["District", "Hospital_count"]
        ],
        on="District",
        how="left"
    )

    assert_no_row_explosion(
        before_rows,
        len(sat),
        "Hospital merge"
    )

    summarize_merge(sat, "Hospital merge")

else:
    print("  ⚠ Hospital dataset not found — skipping")


print("\n────────────────────────────────────────────────────")
print("Final sorting and cleaning")

sat = sat.sort_values(
    by=["District", "Year", "Month"]
).reset_index(drop=True)


out = os.path.join(
    OUTPUT,
    "master_features.csv"
)

sat.to_csv(out, index=False)


print("\n════════════════════════════════════════════════════")
print("  MASTER FEATURE DATASET CREATED")
print("════════════════════════════════════════════════════")

print(f"\n✓ Saved:")
print(f"  {out}")

print(f"\n✓ Final shape:")
print(f"  {sat.shape[0]} rows × {sat.shape[1]} columns")

print(f"\n✓ Districts:")
print(f"  {sat['District'].nunique()}")

print(f"\n✓ Year range:")
print(f"  {sat['Year'].min()}–{sat['Year'].max()}")

print(f"\n✓ Columns ({len(sat.columns)}):")

for c in sat.columns:
    miss = sat[c].isna().sum()
    pct = (miss / len(sat)) * 100

    print(
        f"  {c:<28} "
        f"missing={miss:<6} "
        f"({pct:5.1f}%)"
    )

print("\n✓ Merge pipeline completed successfully")
print("✓ Ready for POST-MERGE TEMPORAL QC")