
import pandas as pd
import numpy as np
import os, sys
from datetime import datetime

RAW = os.path.join("data", "raw")
QC  = os.path.join("outputs", "qc")
os.makedirs(QC, exist_ok=True)

REPORT_LINES = []
def log(line="", color=None):
    REPORT_LINES.append(line)
    print(line)

def section(title):
    log()
    log("═" * 65)
    log(f"  {title}")
    log("═" * 65)

def ok(msg):   log(f"  ✓  {msg}")
def warn(msg): log(f"  ⚠  {msg}")
def fail(msg): log(f"  ✗  {msg}")


CANONICAL_DISTRICTS = {
    
    
    "Simdega": ["Simdega"],
    "East Singhbhum"     : ["East Singhbhum","Purbi Singhbhum","East_Singhbhum",
                             "Purbi_Singhbhum","PurbiSinghbhum","EastSinghbhum","Purba Singhbhum"],
    "West Singhbhum"     : ["West Singhbhum","Pashchimi Singhbhum","West_Singhbhum",
                             "Pashchimi_Singhbhum","PashchimSinghbhum","Pashchim Singhbhum"],
    "Seraikela Kharsawan": ["Saraikela Kharsawan","Saraikela-Kharsawan",
                             "Seraikela Kharsawan","Seraikela_Kharsawan",
                             "Saraikela_Kharsawan"],
    "Ranchi"             : ["Ranchi"],
    "Dhanbad"            : ["Dhanbad"],
    "Bokaro"             : ["Bokaro"],
    "Dumka"              : ["Dumka"],
    "Hazaribagh"         : ["Hazaribagh","Hazaribag"],
    "Deoghar"            : ["Deoghar"],
    "Giridih"            : ["Giridih"],
    "Pakur"              : ["Pakur"],
    "Sahibganj"          : ["Sahibganj","Sahebganj","Sahib Ganj"],
    "Chatra"             : ["Chatra"],
    "Lohardaga"          : ["Lohardaga"],
    "Godda"              : ["Godda"],
    "Gumla"              : ["Gumla"],
    "Latehar"            : ["Latehar"],
    "Palamu"             : ["Palamu","Palamau"],
    "Ramgarh"            : ["Ramgarh"],
    "Jamtara"            : ["Jamtara"],
    "Garhwa"             : ["Garhwa"],
    "Koderma"            : ["Koderma"],
}

ALIAS_TO_CANONICAL = {}
for canonical, aliases in CANONICAL_DISTRICTS.items():
    for alias in aliases:
        ALIAS_TO_CANONICAL[alias.strip()] = canonical

VALID_DISTRICTS = set(CANONICAL_DISTRICTS.keys())

def harmonize_districts(df):
    if "District" not in df.columns:
        return df
    df = df.copy()
    df["District_raw"] = df["District"]
    df["District"] = df["District"].str.strip().map(
        lambda x: ALIAS_TO_CANONICAL.get(x, x)
    )
    unmapped = df[~df["District"].isin(VALID_DISTRICTS)]["District"].unique()
    return df, unmapped


RANGES = {
    "LST_C"           : (-5, 55),
    "LST_Night_C"     : (-5, 45),
    "Rainfall_mm"     : (0, 1500),
    "Humidity_pct"    : (0, 100),
    "NDVI"            : (-0.2, 1.0),
    "mean"            : (-999.1, 9999),
    "SoilMoist_m3m3"  : (0.0, 0.60),
    "Nightlights"     : (0, 500),
    "Elevation_m"     : (0, 2500),
    "Slope_deg"       : (0, 45),
    "Forest_pct"      : (0, 100),
    "Cropland_pct"    : (0, 100),
    "Builtup_pct"     : (0, 100),
    "Pop_density"     : (0, 20000),
    "Water_pct"       : (0, 100),
    "Perm_Water_pct"  : (0, 100),
    "Seas_Water_pct"  : (0, 100),
    "Nightlights"     : (0, 500),
}

INVALID_SENTINELS = [-999, -9999, -9998, 999999]


DATASETS = [
    {
        "key"        : "satellite",
        "file"       : "satellite_climate_district.csv",
        "temporal"   : True,
        "expected_cols": ["District","Year","Month","LST_C","Rainfall_mm","NDVI","Humidity_pct"],
        "desc"       : "Base satellite climate (MODIS+CHIRPS+ERA5+NDVI)",
    },
    {
        "key"        : "night_lst",
        "file"       : "jhk_night_lst_2010_2023.csv",
        "temporal"   : True,
        "expected_cols": ["District","year","month","mean"],
        "rename"     : {"year":"Year","month":"Month","mean":"LST_Night_C"},
        "desc"       : "Night LST (MODIS MOD11A2)",
    },
    {
        "key"        : "jrc_monthly",
        "file"       : "jhk_jrc_monthly_water_2010_2023.csv",
        "temporal"   : True,
        "expected_cols": ["District","year","month","mean"],
        "rename"     : {"year":"Year","month":"Month","mean":"Water_pct"},
        "desc"       : "JRC monthly water presence",
    },
    {
        "key"        : "jrc_static",
        "file"       : "jhk_jrc_static_occurrence.csv",
        "temporal"   : False,
        "expected_cols": ["District","Perm_Water_pct","Seas_Water_pct"],
        "desc"       : "JRC permanent + seasonal water %",
    },
    {
        "key"        : "smap",
        "file"       : "jhk_smap_2015_2023.csv",
        "temporal"   : True,
        "expected_cols": ["District","year","month","mean"],
        "rename"     : {"year":"Year","month":"Month","mean":"SoilMoist_m3m3"},
        "desc"       : "SMAP soil moisture (Apr 2015 onwards)",
    },
    {
        "key"        : "nightlights",
        "file"       : "jhk_nightlights_2012_2023.csv",
        "temporal"   : True,
        "time_res"   : "annual",
        "expected_cols": ["District","year","mean"],
        "rename"     : {"year":"Year","mean":"Nightlights"},
        "desc"       : "VIIRS nighttime lights (2012 onwards)",
    },
    {
        "key"        : "static",
        "file"       : "jhk_static_features.csv",
        "temporal"   : False,
        "expected_cols": ["District","Elevation_m","Slope_deg","Pop_density"],
        "desc"       : "Static terrain + LULC + population",
    },
]


def check_shapes(loaded):
    section("CHECK 1 — DATASET SHAPES AND COLUMNS")
    for d in DATASETS:
        key  = d["key"]
        desc = d["desc"]
        if key not in loaded:
            fail(f"{d['file']} — FILE NOT FOUND")
            continue

        df   = loaded[key]
        log(f"\n  [{key}] {desc}")
        log(f"    File  : {d['file']}")
        log(f"    Shape : {df.shape[0]} rows × {df.shape[1]} cols")
        log(f"    Cols  : {list(df.columns)}")

        missing_expected = [c for c in d["expected_cols"] if c not in df.columns]
        if missing_expected:
            fail(f"Missing expected columns: {missing_expected}")
        else:
            ok(f"All expected columns present")


def check_districts(loaded):
    section("CHECK 2 — DISTRICT NAME HARMONIZATION")
    mapping_records = []

    for d in DATASETS:
        key = d["key"]
        if key not in loaded: continue
        df = loaded[key]
        if "District" not in df.columns: continue

        raw_names = df["District"].str.strip().unique()
        log(f"\n  [{key}]")

        for name in sorted(raw_names):
            canonical = ALIAS_TO_CANONICAL.get(name, None)
            if canonical:
                status = "OK" if canonical == name else f"REMAP → {canonical}"
                mapping_records.append({"Dataset":key,"Raw":name,"Canonical":canonical,"Status":status})
                if canonical != name:
                    warn(f"'{name}' → '{canonical}'")
            else:
                fail(f"UNMAPPED: '{name}' — add to CANONICAL_DISTRICTS mapping")
                mapping_records.append({"Dataset":key,"Raw":name,"Canonical":"UNMAPPED","Status":"UNMAPPED"})

        n_valid = sum(1 for n in raw_names if ALIAS_TO_CANONICAL.get(n) in VALID_DISTRICTS)
        log(f"    Valid districts: {n_valid} / {len(raw_names)}")
        if n_valid == len(raw_names):
            ok(f"All district names mappable")

    mapping_df = pd.DataFrame(mapping_records)
    path = os.path.join(QC, "district_mapping.csv")
    mapping_df.to_csv(path, index=False)
    ok(f"District mapping saved: {path}")
    return mapping_df


def check_duplicates(loaded):
    section("CHECK 3 — DUPLICATE ROW DETECTION")

    for d in DATASETS:
        key = d["key"]
        if key not in loaded: continue
        df  = loaded[key]

        if d["temporal"]:
            time_res = d.get("time_res", "monthly")
            if time_res == "annual":
                key_cols = [c for c in ["District","year","Year"] if c in df.columns][:2]
            else:
                year_col  = "Year" if "Year" in df.columns else "year"
                month_col = "Month" if "Month" in df.columns else "month"
                key_cols  = [c for c in ["District", year_col, month_col] if c in df.columns]
        else:
            key_cols = ["District"]

        key_cols = list(dict.fromkeys(key_cols))
        dups = df[df.duplicated(subset=key_cols, keep=False)]

        log(f"\n  [{key}]  key={key_cols}")
        if len(dups) == 0:
            ok(f"No duplicates found")
        else:
            fail(f"{len(dups)} duplicate rows found!")
            log(dups.head(5).to_string(index=False))


def check_missing(loaded):
    section("CHECK 4 — MISSING VALUES + SENTINEL VALUES (-999 etc.)")
    miss_records = []
    invalid_records = []

    for d in DATASETS:
        key = d["key"]
        if key not in loaded: continue
        df  = loaded[key].copy()
        log(f"\n  [{key}]")

        if "rename" in d:
            df = df.rename(columns=d["rename"])

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            n_total = len(df)
            n_nan   = df[col].isna().sum()

            n_sent  = sum(df[col].isin(INVALID_SENTINELS))

            df_clean = df[col].replace(INVALID_SENTINELS, np.nan)
            n_total_miss = df_clean.isna().sum()
            miss_pct = n_total_miss / n_total * 100

            miss_records.append({
                "Dataset"     : key,
                "Column"      : col,
                "Total_rows"  : n_total,
                "NaN_count"   : n_nan,
                "Sentinel_count": n_sent,
                "Total_missing": n_total_miss,
                "Missing_pct" : round(miss_pct, 1),
            })

            if n_sent > 0:
                fail(f"{col}: {n_sent} sentinel values (-999 etc.) — must replace with NaN")
                invalid_records.append({
                    "Dataset": key, "Column": col, "Issue": "sentinel",
                    "Count": n_sent, "Values": "-999/-9999"
                })
            elif miss_pct > 50:
                warn(f"{col}: {miss_pct:.1f}% missing — check if expected")
            elif miss_pct > 0:
                log(f"    {col:<25} {miss_pct:5.1f}% missing  (NaN={n_nan}, sentinel={n_sent})")
            else:
                ok(f"{col}: complete (0% missing)")

    miss_df    = pd.DataFrame(miss_records)
    invalid_df = pd.DataFrame(invalid_records)

    miss_df.to_csv(os.path.join(QC, "missingness.csv"), index=False)
    ok(f"Missingness report saved: outputs/qc/missingness.csv")

    if len(invalid_df) > 0:
        invalid_df.to_csv(os.path.join(QC, "invalid_values.csv"), index=False)
        fail(f"Invalid values found — see outputs/qc/invalid_values.csv")
    else:
        ok("No sentinel values found across all datasets")

    return miss_df


def check_ranges(loaded):
    section("CHECK 5 — PHYSICAL RANGE VALIDATION")

    for d in DATASETS:
        key = d["key"]
        if key not in loaded: continue
        df  = loaded[key].copy()
        if "rename" in d:
            df = df.rename(columns=d["rename"])

        log(f"\n  [{key}]")
        any_issue = False

        for col, (lo, hi) in RANGES.items():
            if col not in df.columns: continue
            vals = df[col].replace(INVALID_SENTINELS, np.nan).dropna()
            if len(vals) == 0: continue

            n_lo  = (vals < lo).sum()
            n_hi  = (vals > hi).sum()
            vmin  = vals.min()
            vmax  = vals.max()
            vmean = vals.mean()

            if n_lo > 0 or n_hi > 0:
                fail(f"{col}: {n_lo} below {lo}, {n_hi} above {hi}  "
                     f"(actual range: {vmin:.2f}–{vmax:.2f})")
                any_issue = True
            else:
                ok(f"{col}: range OK  [{vmin:.2f} – {vmax:.2f}]  mean={vmean:.2f}")

        if not any_issue:
            ok("All numeric ranges valid")


def check_temporal(loaded):
    section("CHECK 6 — TEMPORAL CONTINUITY")
    gap_records = []

    for d in DATASETS:
        key = d["key"]
        if key not in loaded: continue
        if not d.get("temporal", False): continue
        if d.get("time_res", "monthly") != "monthly": continue

        df = loaded[key].copy()
        if "rename" in d:
            df = df.rename(columns=d["rename"])

        year_col  = "Year"  if "Year"  in df.columns else "year"
        month_col = "Month" if "Month" in df.columns else "month"

        if "District" not in df.columns: continue
        if year_col not in df.columns: continue
        if month_col not in df.columns: continue

        df[year_col]  = pd.to_numeric(df[year_col],  errors="coerce")
        df[month_col] = pd.to_numeric(df[month_col], errors="coerce")

        log(f"\n  [{key}]")
        yr_min = int(df[year_col].min())
        yr_max = int(df[year_col].max())
        log(f"    Year range: {yr_min}–{yr_max}")

        districts = df["District"].str.strip().unique()
        expected  = pd.MultiIndex.from_product(
            [districts, range(yr_min, yr_max+1), range(1, 13)],
            names=["District", year_col, month_col]
        )
        actual = pd.MultiIndex.from_arrays([
            df["District"].str.strip(), df[year_col], df[month_col]
        ])
        missing_idx = expected.difference(actual)

        if len(missing_idx) == 0:
            ok(f"Full temporal coverage — no gaps")
        else:
            warn(f"{len(missing_idx)} District×Year×Month combinations missing")
            gaps_df = pd.DataFrame(list(missing_idx), columns=["District", year_col, month_col])
            gap_records.append(gaps_df.assign(Dataset=key))
            log(gaps_df.head(10).to_string(index=False))

    if gap_records:
        all_gaps = pd.concat(gap_records)
        all_gaps.to_csv(os.path.join(QC, "temporal_gaps.csv"), index=False)
        warn(f"Temporal gaps saved: outputs/qc/temporal_gaps.csv")
    else:
        ok("No temporal gaps across any dataset")


def check_spatial(loaded):
    section("CHECK 7 — SPATIAL SANITY CHECK")

    key = "satellite"
    if key not in loaded:
        warn("satellite_climate_district.csv not loaded — skipping spatial check")
        return

    df = loaded[key].copy()
    if "District" not in df.columns: return

    numeric_cols = ["LST_C","Rainfall_mm","NDVI","Humidity_pct"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    log(f"\n  Checking spatial variation across districts...")
    log(f"  (Ranchi and Palamu should look different — if they don't, data is suspect)")

    for col in numeric_cols:
        district_means = df.groupby("District")[col].mean().dropna()
        cv = district_means.std() / district_means.mean() * 100

        if cv < 2:
            fail(f"{col}: CV={cv:.1f}% — almost NO spatial variation! Likely Ranchi proxy issue")
        elif cv < 8:
            warn(f"{col}: CV={cv:.1f}% — low spatial variation (expected > 8%)")
        else:
            ok(f"{col}: CV={cv:.1f}% — good spatial variation")

        top3    = district_means.nlargest(3)
        bottom3 = district_means.nsmallest(3)
        log(f"    Highest: {dict(zip(top3.index, top3.round(2).values))}")
        log(f"    Lowest:  {dict(zip(bottom3.index, bottom3.round(2).values))}")


def run_qc():
    log("═" * 65)
    log(f"  JHARKHAND DENGUE — PRE-MERGE QC REPORT")
    log(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log("═" * 65)

    section("LOADING DATASETS")
    loaded = {}
    for d in DATASETS:
        path = os.path.join(RAW, d["file"])
        if os.path.exists(path):
            df = pd.read_csv(path)
            loaded[d["key"]] = df
            ok(f"Loaded [{d['key']}]: {df.shape}  — {d['file']}")
        else:
            warn(f"NOT FOUND: {d['file']}")

    check_shapes(loaded)
    mapping_df = check_districts(loaded)
    check_duplicates(loaded)
    miss_df = check_missing(loaded)
    check_ranges(loaded)
    check_temporal(loaded)
    check_spatial(loaded)

    section("QC SUMMARY — ACTION ITEMS")

    issues = [l for l in REPORT_LINES if l.strip().startswith("✗")]
    warnings = [l for l in REPORT_LINES if l.strip().startswith("⚠")]

    if issues:
        log(f"\n  MUST FIX before merging ({len(issues)} issues):")
        for i in issues: log(f"    {i.strip()}")
    else:
        ok("No critical issues found")

    if warnings:
        log(f"\n  REVIEW before merging ({len(warnings)} warnings):")
        for w in warnings: log(f"    {w.strip()}")

    if not issues and not warnings:
        ok("ALL CHECKS PASSED — safe to run merge_extended_features.py")

    report_path = os.path.join(QC, "qc_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT_LINES))
    log(f"\n  Full report saved: {report_path}")

    return len(issues) == 0


if __name__ == "__main__":
    passed = run_qc()
    if passed:
        print("\n  ✓ QC PASSED — run merge_extended_features.py next")
    else:
        print("\n  ✗ QC FAILED — fix issues above before merging")
        sys.exit(1)