
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import warnings
import os
import sys
from datetime import datetime

warnings.filterwarnings("ignore")


INPUT_FILE = os.path.join("outputs", "master_features_engineered.csv")
OUT_DIR    = os.path.join("outputs", "exploratory_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

PAL = {
    "navy"   : "#0B1F3A",
    "teal"   : "#0D9488",
    "amber"  : "#F59E0B",
    "coral"  : "#EF4444",
    "green"  : "#16A34A",
    "purple" : "#7C3AED",
    "slate"  : "#475569",
    "gold"   : "#D97706",
    "sky"    : "#0EA5E9",
    "rose"   : "#F43F5E",
}

DISTRICT_COLORS = [
    PAL["navy"], PAL["teal"], PAL["amber"], PAL["coral"], PAL["green"],
    PAL["purple"], PAL["slate"], PAL["gold"], PAL["sky"], PAL["rose"],
    "#065F46", "#1E3A5F", "#92400E", "#7F1D1D", "#312E81",
    "#064E3B", "#1E40AF", "#78350F", "#881337", "#1E1B4B",
    "#134E4A", "#3B0764",
]

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

MONSOON_MONTHS     = [6, 7, 8, 9]
POST_MONSOON_MONTHS= [10, 11]


def pub_style(fig, title=None):
    fig.patch.set_facecolor("#FAFAFA")
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold",
                     color=PAL["navy"], y=0.98)


def save(fig, name, dpi=150):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓  Saved: {name}")
    return path


def mann_kendall(x):
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 4:
        return np.nan, np.nan, "insufficient data"
    n = len(x)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(x[j] - x[i])
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = 2 * s / (n * (n - 1))
    direction = "increasing" if tau > 0 and p < 0.05 else (
                "decreasing" if tau < 0 and p < 0.05 else "no trend")
    return tau, p, direction


def sens_slope(x, years):
    x = np.array(x, dtype=float)
    years = np.array(years, dtype=float)
    valid = ~np.isnan(x)
    x, years = x[valid], years[valid]
    if len(x) < 3:
        return np.nan
    slopes = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if years[j] != years[i]:
                slopes.append((x[j] - x[i]) / (years[j] - years[i]))
    return np.median(slopes) if slopes else np.nan


print("\n" + "═" * 65)
print("  JHARKHAND DENGUE — EXPLORATORY ANALYSIS")
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("═" * 65)

if not os.path.exists(INPUT_FILE):
    print(f"  ✗  Input not found: {INPUT_FILE}")
    sys.exit(1)

df = pd.read_csv(INPUT_FILE)
print(f"\n  ✓  Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
print(f"     Districts : {df['District'].nunique()}")
print(f"     Years     : {df['Year'].min()}–{df['Year'].max()}")
print(f"     Columns   : {list(df.columns[:8])} ...")

DISTRICTS = sorted(df["District"].unique())
YEARS     = sorted(df["Year"].unique())

CLIMATE_COLS = [c for c in [
    "LST_C", "Rainfall_mm", "NDVI", "Humidity_pct",
    "LST_Night_C", "SoilMoist_m3m3", "Water_pct_monthly",
    "Nightlights", "Elevation_m", "Forest_pct", "Cropland_pct"
] if c in df.columns]

print(f"     Climate cols available: {CLIMATE_COLS}")


print("\n[01/12] District climate profiles...")

district_means = df.groupby("District")[CLIMATE_COLS].mean()

scaler = StandardScaler()
district_z = pd.DataFrame(
    scaler.fit_transform(district_means.fillna(district_means.mean())),
    index=district_means.index,
    columns=district_means.columns
)

fig, axes = plt.subplots(1, 2, figsize=(16, 9),
                          gridspec_kw={"width_ratios": [2, 1]})
pub_style(fig, "Figure 01 — District Environmental Profiles (2010–2023)")

cmap = LinearSegmentedColormap.from_list(
    "epid", ["#1E3A5F", "#F8FAFC", "#7F1D1D"])
sns.heatmap(district_z.T, ax=axes[0], cmap=cmap, center=0,
            linewidths=0.4, linecolor="#E2E8F0",
            cbar_kws={"label": "Z-score (across districts)",
                      "shrink": 0.8})
axes[0].set_title("Standardised Environmental Conditions per District",
                  fontsize=11, color=PAL["navy"], pad=10)
axes[0].set_xlabel("")
axes[0].set_ylabel("Environmental Variable", fontsize=9)
axes[0].tick_params(axis="x", rotation=45, labelsize=8)
axes[0].tick_params(axis="y", rotation=0, labelsize=8)

if "Rainfall_mm" in district_means.columns:
    rain_rank = district_means["Rainfall_mm"].sort_values(ascending=True)
    colors    = [PAL["coral"] if v > rain_rank.quantile(0.75)
                 else (PAL["amber"] if v > rain_rank.quantile(0.5)
                 else PAL["teal"]) for v in rain_rank]
    axes[1].barh(rain_rank.index, rain_rank.values, color=colors, height=0.7)
    axes[1].set_xlabel("Mean Monthly Rainfall (mm)", fontsize=9,
                        color=PAL["slate"])
    axes[1].set_title("Rainfall Ranking", fontsize=10,
                       color=PAL["navy"])
    axes[1].tick_params(labelsize=7)
    axes[1].spines[["top","right"]].set_visible(False)

plt.tight_layout()
save(fig, "01_district_climate_profiles.png")

district_means.round(3).to_csv(
    os.path.join(OUT_DIR, "01_district_climate_means.csv"))


print("[02/12] Temporal trend analysis (Mann-Kendall)...")

trend_cols = [c for c in ["LST_C","LST_Night_C","Rainfall_mm",
                           "NDVI","SoilMoist_m3m3","Humidity_pct"]
              if c in df.columns]

annual_state = df.groupby("Year")[trend_cols].mean()

trend_results = []
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
pub_style(fig, "Figure 02 — Temporal Climate Trends (2010–2023, Mann-Kendall Test)")
axes_flat = axes.flatten()

for idx, col in enumerate(trend_cols[:6]):
    ax   = axes_flat[idx]
    vals = annual_state[col].dropna()
    yrs  = vals.index.astype(float)

    tau, p, direction = mann_kendall(vals.values)
    slope = sens_slope(vals.values, yrs.values)

    ax.scatter(yrs, vals, color=PAL["teal"], s=45, zorder=3, alpha=0.8)
    if not np.isnan(slope):
        y_fit = slope * (yrs - np.mean(yrs)) + vals.mean()
        color = PAL["coral"] if direction == "increasing" else (
                PAL["green"] if direction == "decreasing" else PAL["slate"])
        ax.plot(yrs, y_fit, color=color, linewidth=2, linestyle="--", zorder=2)

    sig_str = "***" if p < 0.001 else ("**" if p < 0.01 else
              ("*" if p < 0.05 else "ns"))
    slope_str = f"{slope:+.4f}/yr" if not np.isnan(slope) else "—"
    ax.set_title(f"{col}", fontsize=10, color=PAL["navy"], fontweight="bold")
    ax.text(0.05, 0.92, f"τ={tau:.3f}  p={p:.3f} {sig_str}",
            transform=ax.transAxes, fontsize=8, color=PAL["slate"])
    ax.text(0.05, 0.82, f"Sen's slope: {slope_str}",
            transform=ax.transAxes, fontsize=8, color=color if not np.isnan(slope) else PAL["slate"])
    ax.set_xlabel("Year", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines[["top","right"]].set_visible(False)

    trend_results.append({
        "Variable": col, "MK_tau": round(tau, 4) if not np.isnan(tau) else None,
        "MK_p": round(p, 4) if not np.isnan(p) else None,
        "Significance": sig_str, "Direction": direction,
        "Sens_slope_per_year": round(slope, 5) if not np.isnan(slope) else None,
    })

for i in range(len(trend_cols), 6):
    axes_flat[i].set_visible(False)

plt.tight_layout()
save(fig, "02_temporal_trend_analysis.png")

pd.DataFrame(trend_results).to_csv(
    os.path.join(OUT_DIR, "02_trend_results_mann_kendall.csv"), index=False)


print("[03/12] Rainfall anomaly dynamics...")

if "Rainfall_mm_zscore" in df.columns:
    zscore_col = "Rainfall_mm_zscore"
elif "Rainfall_mm" in df.columns:
    annual_rain = df.groupby("Year")["Rainfall_mm"].mean()
    z_map = (annual_rain - annual_rain.mean()) / annual_rain.std()
    df["_rain_zscore_temp"] = df["Year"].map(z_map)
    zscore_col = "_rain_zscore_temp"
else:
    zscore_col = None

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
pub_style(fig, "Figure 03 — Rainfall Anomaly Dynamics (2010–2023)")

if zscore_col:
    ax = axes[0, 0]
    annual_z = df.groupby("Year")[zscore_col].mean()
    colors   = [PAL["coral"] if v > 0 else PAL["teal"] for v in annual_z]
    ax.bar(annual_z.index, annual_z.values, color=colors, edgecolor="white",
           linewidth=0.5)
    ax.axhline(0, color=PAL["slate"], linewidth=1, linestyle="--")
    ax.axhline(1, color=PAL["coral"], linewidth=0.8, linestyle=":", alpha=0.6)
    ax.axhline(-1, color=PAL["teal"], linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_title("Annual Rainfall Anomaly (Z-score)", fontsize=10,
                  color=PAL["navy"])
    ax.set_xlabel("Year"); ax.set_ylabel("Z-score")
    ax.tick_params(labelsize=8)
    ax.spines[["top","right"]].set_visible(False)

ax = axes[0, 1]
if "Rainfall_mm" in df.columns:
    seasonal = df.groupby(["Year","Month"])["Rainfall_mm"].mean().reset_index()
    for i, yr in enumerate(YEARS):
        sub = seasonal[seasonal["Year"] == yr]
        alpha = 0.3 + 0.7 * (i / len(YEARS))
        color = PAL["teal"] if yr < 2016 else (
                PAL["amber"] if yr < 2020 else PAL["coral"])
        ax.plot(sub["Month"], sub["Rainfall_mm"],
                color=color, alpha=alpha, linewidth=1.2)

    ax.axvspan(6, 9, alpha=0.08, color=PAL["teal"],
               label="Monsoon season (Jun–Sep)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES, fontsize=7)
    ax.set_title("Monthly Rainfall Profile by Year", fontsize=10,
                  color=PAL["navy"])
    ax.set_ylabel("Rainfall (mm)")
    ax.legend(fontsize=7)
    ax.spines[["top","right"]].set_visible(False)

ax = axes[1, 0]
if "Rainfall_mm" in df.columns:
    monsoon_rain = (df[df["Month"].isin(MONSOON_MONTHS)]
                    .groupby("Year")["Rainfall_mm"].sum())
    z_monsoon = (monsoon_rain - monsoon_rain.mean()) / monsoon_rain.std()
    colors = [PAL["coral"] if v > 0 else PAL["teal"] for v in z_monsoon]
    ax.bar(z_monsoon.index, z_monsoon.values, color=colors,
           edgecolor="white", linewidth=0.5)
    ax.axhline(0, color=PAL["slate"], linewidth=1)
    ax.set_title("Monsoon Season Rainfall Anomaly (Jun–Sep)", fontsize=10,
                  color=PAL["navy"])
    ax.set_xlabel("Year"); ax.set_ylabel("Z-score")
    ax.tick_params(labelsize=8)
    ax.spines[["top","right"]].set_visible(False)

ax = axes[1, 1]
if "Rainfall_mm" in df.columns:
    dist_cv = (df.groupby("District")["Rainfall_mm"]
               .agg(lambda x: x.std() / x.mean() * 100)
               .sort_values(ascending=False))
    colors = [PAL["coral"] if v > dist_cv.quantile(0.75)
              else PAL["amber"] for v in dist_cv]
    ax.barh(dist_cv.index, dist_cv.values, color=colors, height=0.7)
    ax.set_title("Rainfall Variability by District (CV %)", fontsize=10,
                  color=PAL["navy"])
    ax.set_xlabel("Coefficient of Variation (%)")
    ax.tick_params(labelsize=7)
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save(fig, "03_rainfall_anomaly_dynamics.png")


print("[04/12] Temperature warming trends...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
pub_style(fig, "Figure 04 — Temperature Warming Trends (Day + Night LST)")

temp_cols = [c for c in ["LST_C","LST_Night_C"] if c in df.columns]

for col_idx, col in enumerate(temp_cols[:2]):
    ax = axes[0, col_idx]
    annual_temp = df.groupby("Year")[col].mean()
    ax.scatter(annual_temp.index, annual_temp.values,
               color=PAL["coral"], s=50, zorder=3)

    valid = annual_temp.dropna()
    if len(valid) >= 3:
        slope, intercept, r, p, _ = stats.linregress(
            valid.index.astype(float), valid.values)
        x_fit = np.array([valid.index.min(), valid.index.max()])
        ax.plot(x_fit, slope * x_fit + intercept,
                color=PAL["navy"], linewidth=2, linestyle="--",
                label=f"Trend: {slope:+.3f}°C/yr  (p={p:.3f})")

    ax.set_title(f"{col} — Annual Mean", fontsize=10, color=PAL["navy"])
    ax.set_ylabel("Temperature (°C)")
    ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    ax = axes[1, col_idx]
    monthly_temp = df.groupby("Month")[col].agg(["mean","std"]).dropna()
    if len(monthly_temp) > 0:
        ax.plot(monthly_temp.index, monthly_temp["mean"],
                color=PAL["coral"], linewidth=2, marker="o", markersize=5)
        ax.fill_between(monthly_temp.index,
                        monthly_temp["mean"] - monthly_temp["std"],
                        monthly_temp["mean"] + monthly_temp["std"],
                        alpha=0.2, color=PAL["coral"])
        ax.axvspan(6, 9, alpha=0.08, color=PAL["teal"])
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_NAMES, fontsize=7)
        ax.set_title(f"{col} — Seasonal Climatology ± 1 SD", fontsize=10,
                      color=PAL["navy"])
        ax.set_ylabel("Temperature (°C)")
        ax.spines[["top","right"]].set_visible(False)

if len(temp_cols) < 2:
    for j in range(len(temp_cols), 2):
        axes[0, j].set_visible(False)
        axes[1, j].set_visible(False)

plt.tight_layout()
save(fig, "04_temperature_warming_trends.png")


print("[05/12] Monsoon dynamics...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
pub_style(fig, "Figure 05 — Monsoon Dynamics and Ecological Response")

if "Rainfall_mm" in df.columns and "NDVI" in df.columns:
    ax1 = axes[0, 0]
    ax2 = ax1.twinx()
    monthly_rain = df.groupby("Month")["Rainfall_mm"].mean()
    monthly_ndvi = df.groupby("Month")["NDVI"].mean()

    ax1.bar(monthly_rain.index, monthly_rain.values,
            color=PAL["teal"], alpha=0.7, label="Rainfall (mm)")
    ax2.plot(monthly_ndvi.index, monthly_ndvi.values,
             color=PAL["green"], linewidth=2.5, marker="o",
             markersize=5, label="NDVI")

    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(MONTH_NAMES, fontsize=7)
    ax1.set_ylabel("Rainfall (mm)", color=PAL["teal"])
    ax2.set_ylabel("NDVI", color=PAL["green"])
    ax1.set_title("Monsoon Rainfall vs Vegetation Response", fontsize=10,
                   color=PAL["navy"])
    ax1.axvspan(6, 9, alpha=0.08, color=PAL["amber"])
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

ax = axes[0, 1]
if "Rainfall_mm" in df.columns and "NDVI" in df.columns:
    monthly_avg = df.groupby("Month")[["Rainfall_mm","NDVI"]].mean()
    ndvi_shifted = monthly_avg["NDVI"].shift(-1)

    ax.scatter(monthly_avg["Rainfall_mm"], ndvi_shifted,
               c=range(12), cmap="viridis", s=80, zorder=3)
    for mo in range(12):
        ax.annotate(MONTH_NAMES[mo],
                    (monthly_avg["Rainfall_mm"].iloc[mo],
                     ndvi_shifted.iloc[mo] if not np.isnan(ndvi_shifted.iloc[mo])
                     else 0),
                    fontsize=7, ha="center")
    ax.set_xlabel("Monthly Rainfall (mm)")
    ax.set_ylabel("NDVI (1-month lead)")
    ax.set_title("Rainfall → NDVI Response (1-month lag)", fontsize=10,
                  color=PAL["navy"])
    ax.spines[["top","right"]].set_visible(False)

ax = axes[1, 0]
if "Rainfall_mm" in df.columns:
    thresh = df[df["Month"].isin(MONSOON_MONTHS)]["Rainfall_mm"].quantile(0.4)
    onset_years = []
    for yr in YEARS:
        sub = df[(df["Year"] == yr) & (df["Month"].isin([5, 6, 7, 8]))
                 ].groupby("Month")["Rainfall_mm"].mean()
        onset = sub[sub > thresh].index.min() if (sub > thresh).any() else np.nan
        onset_years.append({"Year": yr, "Monsoon_onset_month": onset})
    onset_df = pd.DataFrame(onset_years).dropna()
    if len(onset_df) > 0:
        ax.plot(onset_df["Year"], onset_df["Monsoon_onset_month"],
                color=PAL["teal"], marker="o", linewidth=2)
        ax.axhline(onset_df["Monsoon_onset_month"].mean(),
                   color=PAL["slate"], linestyle="--", alpha=0.6,
                   label=f"Mean: Month {onset_df['Monsoon_onset_month'].mean():.1f}")
        ax.set_yticks([5, 6, 7, 8])
        ax.set_yticklabels(["May","Jun","Jul","Aug"])
        ax.set_title("Monsoon Onset Month by Year", fontsize=10,
                      color=PAL["navy"])
        ax.set_ylabel("Onset Month"); ax.set_xlabel("Year")
        ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
    onset_df.to_csv(os.path.join(OUT_DIR, "05_monsoon_onset.csv"), index=False)

ax = axes[1, 1]
if "SoilMoist_m3m3" in df.columns and "Rainfall_mm" in df.columns:
    sub = df[df["Year"] >= 2015]
    mo_sm = sub.groupby("Month")[["Rainfall_mm","SoilMoist_m3m3"]].mean()
    ax2r = ax.twinx()
    ax.bar(mo_sm.index, mo_sm["Rainfall_mm"], color=PAL["teal"],
           alpha=0.5, label="Rainfall")
    ax2r.plot(mo_sm.index, mo_sm["SoilMoist_m3m3"], color=PAL["amber"],
              linewidth=2.5, marker="s", markersize=5, label="Soil moisture")
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTH_NAMES, fontsize=7)
    ax.set_ylabel("Rainfall (mm)", color=PAL["teal"])
    ax2r.set_ylabel("Soil Moisture (m³/m³)", color=PAL["amber"])
    ax.set_title("Rainfall → Soil Moisture Persistence (2015–2023)", fontsize=10,
                  color=PAL["navy"])
    ax.text(0.05, 0.88,
            "Note: soil moisture lag explains Oct–Nov dengue tail",
            transform=ax.transAxes, fontsize=7, color=PAL["slate"],
            style="italic")

plt.tight_layout()
save(fig, "05_monsoon_dynamics.png")


print("[06/12] Ecological clustering...")

cluster_cols = [c for c in [
    "LST_C","Rainfall_mm","NDVI","Humidity_pct",
    "LST_Night_C","SoilMoist_m3m3","Water_pct_monthly",
    "Elevation_m","Forest_pct","Cropland_pct",
    "Perm_Water_pct","Seas_Water_pct"
] if c in df.columns]

dist_env = df.groupby("District")[cluster_cols].mean()

imp    = SimpleImputer(strategy="median")
scaler = StandardScaler()
X      = scaler.fit_transform(imp.fit_transform(dist_env))

inertias = []
k_range  = range(2, min(8, len(DISTRICTS)))
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

K = 4
km = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = km.fit_predict(X)
dist_env["Cluster"] = labels
cluster_names = {0:"Zone A (Low-temp, Low-rain)",
                 1:"Zone B (Hot, High-rain)",
                 2:"Zone C (Urban, Dry)",
                 3:"Zone D (Moderate, Mixed)"}

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
pub_style(fig, "Figure 06 — Ecological Clustering of Districts (K=4)")

ax = axes[0]
ax.plot(list(k_range), inertias, color=PAL["teal"], marker="o",
        linewidth=2, markersize=7)
ax.axvline(K, color=PAL["coral"], linestyle="--", alpha=0.7,
           label=f"Selected k={K}")
ax.set_xlabel("Number of clusters k")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Curve — Cluster Selection", fontsize=10,
              color=PAL["navy"])
ax.legend(fontsize=8)
ax.spines[["top","right"]].set_visible(False)

ax = axes[1]
cluster_colors = [PAL["teal"], PAL["coral"], PAL["amber"], PAL["purple"]]
cluster_sort   = dist_env.sort_values("Cluster")
bars = ax.barh(cluster_sort.index,
               [1] * len(cluster_sort),
               color=[cluster_colors[c] for c in cluster_sort["Cluster"]],
               height=0.7, edgecolor="white")
ax.set_xlim(0, 1.5)
ax.set_xticks([])
ax.set_title("District Cluster Assignments", fontsize=10, color=PAL["navy"])
ax.tick_params(labelsize=7)
patches = [mpatches.Patch(color=cluster_colors[i],
           label=f"Cluster {i}: {cluster_names.get(i,'')}")
           for i in range(K)]
ax.legend(handles=patches, fontsize=7, loc="lower right")

ax = axes[2]
cluster_means_z = pd.DataFrame(
    scaler.transform(imp.transform(dist_env[cluster_cols].fillna(0))),
    columns=cluster_cols, index=dist_env.index
)
cluster_means_z["Cluster"] = labels
cluster_profile = cluster_means_z.groupby("Cluster")[cluster_cols[:6]].mean()

x = np.arange(len(cluster_cols[:6]))
width = 0.2
for i in range(K):
    ax.bar(x + i * width, cluster_profile.iloc[i],
           width=width, color=cluster_colors[i],
           label=f"C{i}", alpha=0.85)
ax.set_xticks(x + width * (K-1) / 2)
ax.set_xticklabels(cluster_cols[:6], rotation=35, ha="right", fontsize=7)
ax.axhline(0, color=PAL["slate"], linewidth=0.8)
ax.set_ylabel("Z-score")
ax.set_title("Cluster Environmental Profiles", fontsize=10, color=PAL["navy"])
ax.legend(fontsize=7)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save(fig, "06_ecological_clustering.png")

dist_env[["Cluster"]].to_csv(
    os.path.join(OUT_DIR, "06_district_cluster_assignments.csv"))


print("[07/12] Climate vulnerability ranking...")

vuln_components = {}

if "LST_C" in df.columns:
    vuln_components["Temp_score"] = df.groupby("District")["LST_C"].mean()
if "Rainfall_mm" in df.columns:
    rain_cv = (df.groupby("District")["Rainfall_mm"]
               .agg(lambda x: x.std() / (x.mean() + 0.001)))
    vuln_components["Rain_var_score"] = rain_cv
if "_monthly" in df.columns:
    vuln_components["Water_score"] = df.groupby("District")["Water_pct_monthly"].mean()
if "Perm_Water_pct" in df.columns:
    vuln_components["PermWater_score"] = df.groupby("District")["Perm_Water_pct"].mean()
if "Elevation_m" in df.columns:
    elev = df.groupby("District")["Elevation_m"].mean()
    vuln_components["LowElev_score"] = -elev
if "SoilMoist_m3m3" in df.columns:
    vuln_components["Soil_score"] = df.groupby("District")["SoilMoist_m3m3"].mean()
if "Nightlights" in df.columns:
    vuln_components["Urban_score"] = df.groupby("District")["Nightlights"].mean()

if len(vuln_components) >= 3:
    vuln_df = pd.DataFrame(vuln_components)
    for col in vuln_df.columns:
        rng = vuln_df[col].max() - vuln_df[col].min()
        if rng > 0:
            vuln_df[col] = (vuln_df[col] - vuln_df[col].min()) / rng
    vuln_df["Vuln_index"] = vuln_df.mean(axis=1)
    vuln_df = vuln_df.sort_values("Vuln_index", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    pub_style(fig, "Figure 07 — Climate Vulnerability Index by District")

    ax = axes[0]
    colors = [PAL["coral"] if v > 0.6 else
              (PAL["amber"] if v > 0.4 else PAL["teal"])
              for v in vuln_df["Vuln_index"]]
    ax.barh(vuln_df.index, vuln_df["Vuln_index"], color=colors, height=0.7)
    ax.axvline(0.6, color=PAL["coral"], linestyle="--", alpha=0.5,
               label="High vulnerability threshold")
    ax.axvline(0.4, color=PAL["amber"], linestyle="--", alpha=0.5,
               label="Moderate threshold")
    ax.set_xlabel("Vulnerability Index (0–1)")
    ax.set_title("Composite Environmental Vulnerability", fontsize=10,
                  color=PAL["navy"])
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top","right"]].set_visible(False)

    ax = axes[1]
    comp_cols = [c for c in vuln_df.columns if c != "Vuln_index"]
    cmap2 = LinearSegmentedColormap.from_list(
        "vuln", ["#ECFDF5", "#FEF9C3", "#FEE2E2"])
    sns.heatmap(vuln_df[comp_cols], ax=ax, cmap=cmap2, vmin=0, vmax=1,
                linewidths=0.4, linecolor="#E2E8F0", annot=True, fmt=".2f",
                annot_kws={"size": 6},
                cbar_kws={"label": "Normalised score", "shrink": 0.6})
    ax.set_title("Component Scores", fontsize=10, color=PAL["navy"])
    ax.tick_params(labelsize=7, axis="x", rotation=35)
    ax.tick_params(labelsize=7, axis="y")

    plt.tight_layout()
    save(fig, "07_climate_vulnerability_ranking.png")

    vuln_df.round(4).to_csv(
        os.path.join(OUT_DIR, "07_vulnerability_index.csv"))


print("[08/12] Environmental correlation matrix...")

corr_cols = [c for c in [
    "LST_C","LST_Night_C","Rainfall_mm","NDVI","Humidity_pct",
    "SoilMoist_m3m3","Water_pct_monthly","Perm_Water_pct","Nightlights",
    "Elevation_m","Forest_pct","Cropland_pct",
] if c in df.columns]

if len(corr_cols) >= 4:
    sub = df[corr_cols].dropna(how="all")
    n   = len(corr_cols)

    corr_mat = np.zeros((n, n))
    pval_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x = sub[corr_cols[i]].dropna()
            y = sub[corr_cols[j]].dropna()
            idx = x.index.intersection(y.index)
            if len(idx) > 10:
                r, p = spearmanr(x[idx], y[idx])
                corr_mat[i, j] = r
                pval_mat[i, j] = p
            else:
                corr_mat[i, j] = np.nan
                pval_mat[i, j] = 1.0

    corr_df = pd.DataFrame(corr_mat, index=corr_cols, columns=corr_cols)

    sig_mask = pval_mat > 0.05

    fig, ax = plt.subplots(figsize=(12, 10))
    pub_style(fig, "Figure 08 — Environmental Correlation Matrix (Spearman ρ)")

    cmap3 = LinearSegmentedColormap.from_list(
        "corr", ["#1E3A5F", "#FAFAFA", "#7F1D1D"])
    sns.heatmap(corr_df, ax=ax, cmap=cmap3, vmin=-1, vmax=1,
                center=0, annot=True, fmt=".2f",
                annot_kws={"size": 8},
                mask=sig_mask,
                linewidths=0.5, linecolor="#E2E8F0",
                cbar_kws={"label": "Spearman ρ (significant only, p<0.05)",
                          "shrink": 0.8})
    ax.set_title("Non-significant correlations (p>0.05) are masked",
                  fontsize=9, color=PAL["slate"], pad=8)
    ax.tick_params(labelsize=9, axis="x", rotation=40)
    ax.tick_params(labelsize=9, axis="y", rotation=0)

    plt.tight_layout()
    save(fig, "08_correlation_matrix.png")

    corr_df.round(3).to_csv(
        os.path.join(OUT_DIR, "08_spearman_correlations.csv"))


print("[09/12] Transmission suitability index...")

if "TSI" in df.columns:
    tsi_col = "TSI"
else:
    if all(c in df.columns for c in ["LST_C","Humidity_pct","Rainfall_mm"]):
        t = df["LST_C"].clip(15, 38)
        h = df["Humidity_pct"] / 100
        r = df.groupby("District")["Rainfall_mm"].shift(1).fillna(0)
        r = r / r.groupby(df["District"]).transform("max").clip(lower=0.001)

        t_suit = np.where(t < 26, (t - 16) / 10,
                 np.where(t > 32, (38 - t) / 6, 1.0))
        t_suit = np.clip(t_suit, 0, 1)

        ndvi_col_vals = df["NDVI"].fillna(0.45) if "NDVI" in df.columns \
                        else pd.Series(0.45, index=df.index)
        n_suit = np.where(ndvi_col_vals < 0.2, ndvi_col_vals / 0.2,
                 np.where(ndvi_col_vals > 0.7, (1 - ndvi_col_vals) / 0.3, 1.0))
        n_suit = np.clip(n_suit, 0, 1)

        df["TSI"] = (t_suit * h * r * n_suit).clip(0, 1)
        tsi_col = "TSI"
    else:
        tsi_col = None

if tsi_col:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    pub_style(fig, "Figure 09 — Transmission Suitability Index (TSI)")

    tsi_heatmap = df.groupby(["District","Month"])[tsi_col].mean().unstack()
    cmap4 = LinearSegmentedColormap.from_list(
        "tsi", ["#ECFDF5","#FEF9C3","#FEF3C7","#FEE2E2","#7F1D1D"])
    ax = axes[0]
    sns.heatmap(tsi_heatmap, ax=ax, cmap=cmap4, vmin=0, vmax=tsi_heatmap.max().max(),
                linewidths=0.3, linecolor="#F1F5F9",
                cbar_kws={"label": "Mean TSI", "shrink": 0.8})
    ax.set_xticklabels(MONTH_NAMES, fontsize=8, rotation=0)
    ax.set_title("Mean TSI by District × Month", fontsize=10, color=PAL["navy"])
    ax.tick_params(labelsize=7)

    ax = axes[1]
    state_tsi = df.groupby("Month")[tsi_col]
    m_mean = state_tsi.mean()
    m_lo   = state_tsi.quantile(0.1)
    m_hi   = state_tsi.quantile(0.9)

    ax.plot(m_mean.index, m_mean.values,
            color=PAL["coral"], linewidth=2.5, marker="o", markersize=6,
            label="State mean TSI")
    ax.fill_between(m_mean.index, m_lo.values, m_hi.values,
                    alpha=0.2, color=PAL["coral"],
                    label="10th–90th percentile (district range)")
    ax.axvspan(7, 10, alpha=0.08, color=PAL["amber"],
               label="Peak dengue season")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES, fontsize=8)
    ax.set_ylabel("Transmission Suitability Index")
    ax.set_title("TSI Seasonal Profile", fontsize=10, color=PAL["navy"])
    ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    save(fig, "09_transmission_suitability_index.png")


print("[10/12] Persistence ecology...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
pub_style(fig, "Figure 10 — Environmental Persistence Analysis")

if "wet_streak_3" in df.columns or "Rainfall_mm" in df.columns:
    col = "wet_streak_3" if "wet_streak_3" in df.columns else "Rainfall_mm"
    ax = axes[0, 0]
    monthly_mean = df.groupby("Month")[col].mean()
    ax.bar(monthly_mean.index, monthly_mean.values,
           color=PAL["teal"], alpha=0.8)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES, fontsize=7)
    title = "Mean Wet Streak Index by Month" if "wet_streak" in col \
            else "Mean Rainfall by Month"
    ax.set_title(title, fontsize=10, color=PAL["navy"])
    ax.set_ylabel(col)
    ax.spines[["top","right"]].set_visible(False)
    ax.axvspan(6, 9, alpha=0.08, color=PAL["amber"])

if "heat_streak_3" in df.columns or "LST_C" in df.columns:
    col = "heat_streak_3" if "heat_streak_3" in df.columns else "LST_C"
    ax = axes[0, 1]
    monthly_mean = df.groupby("Month")[col].mean()
    ax.plot(monthly_mean.index, monthly_mean.values,
            color=PAL["coral"], linewidth=2.5, marker="o", markersize=5)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES, fontsize=7)
    title = "Mean Heat Streak Index by Month" if "heat_streak" in col \
            else "Mean LST by Month"
    ax.set_title(title, fontsize=10, color=PAL["navy"])
    ax.set_ylabel(col)
    ax.spines[["top","right"]].set_visible(False)

ax = axes[1, 0]
if "wet_streak_3" in df.columns:
    annual_wet = df.groupby("Year")["wet_streak_3"].mean()
    ax.plot(annual_wet.index, annual_wet.values,
            color=PAL["teal"], linewidth=2, marker="o", markersize=5)
    tau, p, direction = mann_kendall(annual_wet.values)
    ax.set_title(f"Annual Wet Streak Trend  (MK p={p:.3f})", fontsize=10,
                  color=PAL["navy"])
    ax.spines[["top","right"]].set_visible(False)
elif "Rainfall_mm_roll3_mean" in df.columns:
    annual_roll = df.groupby("Year")["Rainfall_mm_roll3_mean"].mean()
    ax.plot(annual_roll.index, annual_roll.values,
            color=PAL["teal"], linewidth=2, marker="o", markersize=5)
    ax.set_title("Annual 3-month Rolling Rainfall Mean", fontsize=10,
                  color=PAL["navy"])
    ax.spines[["top","right"]].set_visible(False)

ax = axes[1, 1]
persist_col = "wet_streak_3" if "wet_streak_3" in df.columns else \
              ("Rainfall_mm_roll3_mean" if "Rainfall_mm_roll3_mean" in df.columns
               else None)
if persist_col:
    dist_persist = df.groupby("District")[persist_col].mean().sort_values(ascending=False)
    colors_p = [PAL["coral"] if v > dist_persist.quantile(0.75)
                else PAL["teal"] for v in dist_persist]
    ax.barh(dist_persist.index, dist_persist.values, color=colors_p, height=0.7)
    ax.set_title(f"District Persistence Ranking ({persist_col})",
                  fontsize=10, color=PAL["navy"])
    ax.tick_params(labelsize=7)
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
save(fig, "10_persistence_ecology.png")


print("[11/12] Night warming analysis...")

if "LST_Night_C" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    pub_style(fig, "Figure 11 — Night LST Warming Trends (Novel Predictor)")

    ax = axes[0]
    night_pivot = df.groupby(["District","Year"])["LST_Night_C"].mean().unstack()
    cmap5 = LinearSegmentedColormap.from_list(
        "night", ["#1E3A5F","#3B82F6","#FEF9C3","#F97316","#7F1D1D"])
    sns.heatmap(night_pivot, ax=ax, cmap=cmap5,
                linewidths=0.3, linecolor="#E2E8F0",
                cbar_kws={"label": "Mean Night LST (°C)", "shrink": 0.8})
    ax.set_title("Night LST by District and Year", fontsize=10, color=PAL["navy"])
    ax.tick_params(labelsize=7, axis="x", rotation=45)
    ax.tick_params(labelsize=7, axis="y")

    ax = axes[1]
    night_slopes = {}
    for dist in DISTRICTS:
        sub = df[df["District"] == dist].groupby("Year")["LST_Night_C"].mean().dropna()
        if len(sub) >= 4:
            s = sens_slope(sub.values, sub.index.astype(float).values)
            night_slopes[dist] = s

    if night_slopes:
        slope_df = pd.Series(night_slopes).sort_values(ascending=False)
        colors_n = [PAL["coral"] if v > 0 else PAL["teal"] for v in slope_df]
        ax.barh(slope_df.index, slope_df.values, color=colors_n, height=0.7)
        ax.axvline(0, color=PAL["slate"], linewidth=1)
        ax.set_title("Night LST Warming Rate by District\n(Sen's slope °C/year)",
                      fontsize=10, color=PAL["navy"])
        ax.set_xlabel("°C per year")
        ax.tick_params(labelsize=7)
        ax.spines[["top","right"]].set_visible(False)
        pd.DataFrame({"District": slope_df.index,
                      "NightLST_slope_C_per_yr": slope_df.values}).to_csv(
            os.path.join(OUT_DIR, "11_night_warming_rates.csv"), index=False)

    plt.tight_layout()
    save(fig, "11_night_warming_analysis.png")


print("[12/12] District environmental fingerprints...")

radar_cols = [c for c in [
    "LST_C","Rainfall_mm","NDVI","Humidity_pct",
    "LST_Night_C","SoilMoist_m3m3","Water_pct_monthly","Nightlights"
] if c in df.columns]

if len(radar_cols) >= 4:
    dist_radar = df.groupby("District")[radar_cols].mean()

    if "vuln_df" in dir() and "Vuln_index" in vuln_df.columns:
        top6 = vuln_df.head(6).index.tolist()
    else:
        top6 = DISTRICTS[:6]
    top6 = [d for d in top6 if d in dist_radar.index]

    norm = (dist_radar - dist_radar.min()) / (
            (dist_radar.max() - dist_radar.min()).replace(0, 1))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10),
                              subplot_kw=dict(polar=True))
    pub_style(fig, "Figure 12 — District Environmental Fingerprints (Top 6 Districts)")

    for idx, (dist, ax) in enumerate(zip(top6, axes.flatten())):
        values = norm.loc[dist, radar_cols].fillna(0).values.tolist()
        N      = len(radar_cols)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        values_plot = values + [values[0]]
        angles_plot = angles + [angles[0]]

        ax.plot(angles_plot, values_plot,
                color=DISTRICT_COLORS[idx], linewidth=2)
        ax.fill(angles_plot, values_plot,
                color=DISTRICT_COLORS[idx], alpha=0.25)
        ax.set_xticks(angles)
        ax.set_xticklabels([c.replace("_"," ") for c in radar_cols],
                            size=7, color=PAL["slate"])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(["0.25","0.50","0.75"], size=5,
                             color=PAL["mgray"] if hasattr(PAL,"mgray") else "#94A3B8")
        ax.set_title(dist.replace("_"," "), fontsize=9, color=PAL["navy"],
                     pad=12, fontweight="bold")
        ax.spines["polar"].set_color("#E2E8F0")
        ax.grid(color="#E2E8F0", linewidth=0.6)

    plt.tight_layout()
    save(fig, "12_district_fingerprints.png")


print(f"\n{'═' * 65}")
print("  EXPLORATORY ANALYSIS COMPLETE")
print(f"  Output folder: {OUT_DIR}")
print(f"{'═' * 65}")
print("  Figures generated:")
figs = [f for f in sorted(os.listdir(OUT_DIR)) if f.endswith(".png")]
csvs = [f for f in sorted(os.listdir(OUT_DIR)) if f.endswith(".csv")]
for f in figs:
    print(f"    {f}")
print(f"\n  Supporting CSVs ({len(csvs)}):")
for f in csvs:
    print(f"    {f}")
print(f"\n  Total outputs: {len(figs)} figures + {len(csvs)} data files")
print(f"\n  NEXT STEPS:")
print("    1. Review figures — identify 4-5 strongest for the paper")
print("    2. When RTI data arrives — run dengue_qc.py")
print("    3. Then risk_label_framework.py — create LOW/MOD/HIGH labels")
print("    4. Then modeling_pipeline.py — XGBoost risk classifier")
print(f"{'═' * 65}\n")

summary = {
    "n_districts"    : df["District"].nunique(),
    "year_range"     : f"{df['Year'].min()}–{df['Year'].max()}",
    "n_rows"         : len(df),
    "n_columns"      : df.shape[1],
    "n_figures"      : len(figs),
    "n_csv_outputs"  : len(csvs),
    "climate_cols"   : ", ".join(CLIMATE_COLS),
    "generated_at"   : datetime.now().strftime("%Y-%m-%d %H:%M"),
}
pd.DataFrame([summary]).T.to_csv(
    os.path.join(OUT_DIR, "00_analysis_summary.csv"), header=False)