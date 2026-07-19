import pandas as pd

path = "data/raw/satellite_climate_district.csv"

df = pd.read_csv(path)

df["Humidity_pct"] = (
    df["Humidity_pct"]
    .replace(-1, pd.NA)
)

df.to_csv(path, index=False)

print(" Replaced humidity placeholder -1 with NaN")