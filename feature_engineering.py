
import pandas as pd
import numpy as np
import os


INPUT = os.path.join(
    "outputs",
    "master_features.csv"
)

OUTPUT = os.path.join(
    "outputs",
    "master_features_engineered.csv"
)

print("\n")
print(" FEATURE ENGINEERING")
df = pd.read_csv(INPUT)

print(f"\nLoaded: {df.shape}")


df = df.sort_values(
    by=["District", "Year", "Month"]
).reset_index(drop=True)


df["Date"] = pd.to_datetime(
    dict(
        year=df["Year"],
        month=df["Month"],
        day=1
    )
)


print("\nCreating lag features...")

lag_config = {
    "Rainfall_mm": [1, 2, 3],
    "LST_C": [1, 2],
    "NDVI": [1, 2],
    "SoilMoist_m3m3": [1, 2],
    "Water_pct_monthly": [1]
}

for feature, lags in lag_config.items():

    if feature not in df.columns:
        print(f" Missing feature: {feature}")
        continue

    for lag in lags:

        new_col = f"{feature}_lag{lag}"

        df[new_col] = (
            df.groupby("District")[feature]
            .shift(lag)
        )

        print(f" {new_col}")


print("\nCreating rolling window features...")

rolling_config = {
    "Rainfall_mm": [3, 6],
    "LST_C": [3],
    "NDVI": [3],
    "SoilMoist_m3m3": [3]
}

for feature, windows in rolling_config.items():

    if feature not in df.columns:
        continue

    for w in windows:

        col = f"{feature}_roll{w}_mean"

        df[col] = (
            df.groupby("District")[feature]
            .transform(
                lambda x:
                x.rolling(
                    window=w,
                    min_periods=1
                ).mean()
            )
        )

        print(f" {col}")


print("\nCreating rolling variability features...")

for feature in ["Rainfall_mm", "LST_C"]:

    if feature not in df.columns:
        continue

    col = f"{feature}_roll3_std"

    df[col] = (
        df.groupby("District")[feature]
        .transform(
            lambda x:
            x.rolling(
                window=3,
                min_periods=1
            ).std()
        )
    )

    print(f" {col}")


print("\nCreating seasonal features...")

df["month_sin"] = np.sin(
    2 * np.pi * df["Month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["Month"] / 12
)

df["monsoon_flag"] = (
    df["Month"]
    .isin([6, 7, 8, 9])
    .astype(int)
)

df["post_monsoon_flag"] = (
    df["Month"]
    .isin([10, 11])
    .astype(int)
)

df["summer_flag"] = (
    df["Month"]
    .isin([3, 4, 5])
    .astype(int)
)

print(" Seasonal encodings")


print("\nCreating anomaly features...")

anomaly_features = [
    "Rainfall_mm",
    "LST_C",
    "NDVI"
]

for feature in anomaly_features:

    if feature not in df.columns:
        continue

    mean = (
        df.groupby("District")[feature]
        .transform("mean")
    )

    std = (
        df.groupby("District")[feature]
        .transform("std")
    )

    col = f"{feature}_zscore"

    df[col] = (
        (df[feature] - mean) / std
    )

    print(f" {col}")


print("\nCreating persistence features...")

df["wet_month"] = (
    df["Rainfall_mm"] > 100
).astype(int)

df["wet_streak_3"] = (
    df.groupby("District")["wet_month"]
    .transform(
        lambda x:
        x.rolling(
            3,
            min_periods=1
        ).sum()
    )
)

df["hot_month"] = (
    df["LST_C"] > df["LST_C"].quantile(0.75)
).astype(int)

df["heat_streak_3"] = (
    df.groupby("District")["hot_month"]
    .transform(
        lambda x:
        x.rolling(
            3,
            min_periods=1
        ).sum()
    )
)

print(" wet_streak_3")
print(" heat_streak_3")


print("\nMissingness summary:")

missing = (
    df.isna()
    .sum()
    .sort_values(ascending=False)
)

print(missing.head(20))


df.to_csv(OUTPUT, index=False)

print("\n")
print(" FEATURE ENGINEERING COMPLETE")
print(f"\nSaved:")
print(f"{OUTPUT}")

print(f"\nFinal shape:")
print(df.shape)

print(f"\nTotal columns:")
print(len(df.columns))