import pandas as pd

old = pd.read_csv(
    "data/raw/old_satellite_climate_district.csv"
)

new = pd.read_csv(
    "data/raw/satellite_climate_district.csv"
)

print("\nOLD shape:", old.shape)
print("NEW shape:", new.shape)

humid = old[[
    "District",
    "Year",
    "Month",
    "Humidity_pct"
]].copy()

humid = humid.drop_duplicates(
    subset=["District", "Year", "Month"]
)

merged = new.merge(
    humid,
    on=["District", "Year", "Month"],
    how="left"
)

merged["Humidity_pct"] = (
    merged["Humidity_pct"]
    .fillna(-1)
)

merged.to_csv(
    "data/raw/satellite_climate_district.csv",
    index=False
)

print("\n Humidity restored")