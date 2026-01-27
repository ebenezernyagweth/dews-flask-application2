import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count
import os
from scipy.stats import norm
import re
import seaborn as sns
import plotly.express as px
import glob 
from pathlib import Path

#------------------------------------------------------------------#
# Generating macros/lists with variable names for different models #
#------------------------------------------------------------------#
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
INPUT = os.path.join(PARENT_DIR, "intermediary_datasets")
SHAPE = os.path.join(PARENT_DIR, "shapefiles")
OUTPUT = os.path.join(PARENT_DIR, "results")
DASHBOARD_DATA = os.path.join(
    os.path.dirname(PARENT_DIR),  
    "early_warning_dashboard-main",
    "data"
)

#---------------------------------------------------------------------------

os.chdir(PARENT_DIR) 



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#------------------------------------------------------------------#
#data = pd.read_pickle(os.path.join(INPUT, 
#                                   'complete_ward_level_dataset.pkl'))

# Find latest data file
matches = [p for p in Path(INPUT).glob("complete_ward_level_dataset_*.pkl") if p.is_file()]

if not matches:
    raise FileNotFoundError(f"No dataset found in {INPUT}")
elif len(matches) == 1:
    target = matches[0]
else:
    # If more than one somehow exists, keep the newest by the updated_on token
    MONTH = {m:i for i,m in enumerate(["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])}
    def updated_on_key(p):
        try:
            mon, year = p.stem.rsplit("_updated_on_", 1)[1].split("_")
            return (int(year), MONTH.get(mon.title(), 0))
        except Exception:
            return (-1, -1)
    target = max(matches, key=updated_on_key)
    print(f"Multiple files found; loading: {target.name}")

data = pd.read_pickle(target)
print(f"Loaded: {target}")

data['time_period'] = pd.to_datetime(
    data['year'].astype(str) + '-' + data['month'].astype(str).str.zfill(2),
    format='%Y-%m'
)
data['time_cont_enc'] = data['time_period'].rank(method='dense').astype(int)
#------------------------------------------------------------------#
# Generating macros/lists with variable names #
#------------------------------------------------------------------#
label_encoder_county = LabelEncoder()
label_encoder_livelihood = LabelEncoder()

# Apply label encoding to 'County' and 'LivelihoodZone'
data['County'] = label_encoder_county.fit_transform(data['County'])
data['LivelihoodZone'] = label_encoder_livelihood.fit_transform(data['LivelihoodZone'])


static = [ 'travel_time_to_cities_2015',
 'density_2015','density_2020','delta_2015_2020']

#===========================================
# Smoothing outcome var with gaussian kernel (trend capture)
#===========================================

def smooth_per_group(df, sigma, columns_to_smooth):
    smoothed_df = df.copy()
    window_size = int(3 * sigma) # Size of the Gaussian kernel (3 sigma covers over 99% of the curve)
    
    # Generate Gaussian kernel weights for the past values including current point (one-sided)
    # Note: We generate a kernel for each possible window size to ensure proper normalization at the start of the series
    kernels = {i: norm.pdf(np.arange(i, -1, -1), 0, sigma) for i in range(window_size + 1)}

    for unit_id in df['Ward'].unique():
        unit_data = df[df['Ward'] == unit_id].sort_values(by='time_cont_enc')       
        for col in columns_to_smooth:
            smoothed_col_name = f'{col}_smoothed'
            smoothed_values = np.zeros_like(unit_data[col].values)

            # One-sided Gaussian smoothing
            for i in range(len(unit_data)):
                # Determine the window size (less at the start of the series)
                current_window_size = min(i, window_size) + 1
                kernel = kernels[current_window_size - 1]
                kernel = kernel / kernel.sum()  # Normalize the kernel weights to sum to 1

                smoothed_values[i] = np.dot(unit_data[col].values[i - current_window_size + 1 : i + 1], kernel)
            
            smoothed_df.loc[unit_data.index, smoothed_col_name] = smoothed_values
    
    return smoothed_df


columns_to_smooth = ['wasting', 'wasting_risk']

data = smooth_per_group(data, sigma=3, columns_to_smooth=columns_to_smooth)


for lag in range(1, 13):  # Lags from 1 to 12 months
        # Generate the lagged data and store it in the dictionary
        data[f'wasting_sm_lag_{lag}'] = data.groupby('Ward')['wasting_smoothed'].shift(lag)
        data[f'wasting_sm_risk_lag_{lag}'] = data.groupby('Ward')['wasting_risk_smoothed'].shift(lag)

#==============================
#Getting dynamic variable lags
#==============================
globals_dict = globals()

# Iterate through specific lag numbers
for i in range(1, 13):  # Adjust the range as needed
    lag_pattern = f"lag_{i}"  # Create the exact lag pattern
    globals_dict[f"l{i}"] = [col for col in data if f"{lag_pattern}_" in col or col.endswith(
        lag_pattern)]
# generates lists of variables with the patterm l1, l2,....(on1-month lag, 2-month lag....)
#===========================================
# Dynamically create lists for each rain type
#===========================================
rain_types = ['long_rain', 'short_rain']

for rain_type in rain_types:
    globals()[rain_type] = [var for var in data if f"{rain_type}_max_prev" in var or f"{rain_type}_total_prev" in var or f"{rain_type}_avg_prev" in var]

for rain_type in rain_types:
    globals()[rain_type + '_lag'] = [
        var for var in data if f"{rain_type}_lag" in var]


#------------------------------------------------------------------#
# Generating datasets for each predictive horizon #
#------------------------------------------------------------------#
# Hybrid model #
# Wasting #

hb_m1 = (
    static +
    long_rain + short_rain +
    l1 + l2 + l3 )

hb_m2 = (
    static +
    long_rain + short_rain +
    l2 + l3 + l4 )

hb_m3 = (
    static +
    long_rain + short_rain +
    l3 + l4 + l5 )

#------------------------------


#--------------------------------------------------
num_cpus = os.cpu_count()

def bootstrap_iteration(model,train, testX, ff_m, num_iterations):
    all_predictions = np.zeros((num_iterations, testX.shape[0]))

    def single_bootstrap_iteration(iteration):
        resampled_data = train.groupby('time_cont_enc', group_keys=False).apply(
            lambda x: x.sample(frac=1, replace=True))

        X_sampled = resampled_data[[c for c in trainX.columns if c in ff_m]]
        y_sampled = resampled_data[outcome_var]

        model.fit(X_sampled, y_sampled)
        y_pred = model.predict(testX)

        return y_pred

    # Parallelize the bootstrap iterations
    all_predictions = Parallel(n_jobs=num_cpus)(
        delayed(single_bootstrap_iteration)(i) for i in tqdm(range(num_iterations))
    )

    return np.array(all_predictions)

#---------------------------------------------------------------------
def train_test_split(data_train, data_test, n_test, pred_h):
    i = n_test-36
    t = n_test + pred_h
    print(i, n_test, t)
    
    mask = (data_train['time_cont_enc']<n_test) & (data_train['time_cont_enc']>=i)
    
    return data[mask], data_test[data_test['time_cont_enc'] == t]
#---------------------------------------------------------------------
# Helper function to write the new predictions in a comprehensive time series, and eliminate old files
# If no historic results file is found, it saves a new one with the predictions
# Otherwise, it generates a file with all results combined

def merge_and_replace_predictions(
    results_dir="results",
    model="hb",
    outcome_var="wasting_smoothed",
    horizons=(1, 2, 3),
    delete_inputs=True,
    observed_df: pd.DataFrame = None,   # MUST contain Ward, time_period, outcome_var
    observed_file: str = None,          # or pass a CSV path
    ward_col: str = "Ward",
    date_col: str = "time_period",
):
    """
    Merge new prediction files with historical predictions, handling overlaps intelligently.
    
    Key features:
    - Backfills actual observed values for the last observed month in historical data
    - Keeps future predictions on overlap (they're more recent)
    - Prevents overwriting of backfilled observed data
    """
    # --- Load observed if path provided ---
    if observed_df is None and observed_file is not None:
        observed_df = pd.read_csv(observed_file)

    # --- Prepare last observed date & map (Ward -> observed value) ---
    last_obs_date = None
    obs_last_map = None
    if observed_df is not None:
        if ward_col not in observed_df.columns:
            raise ValueError(f"'{ward_col}' missing in observed data.")
        if date_col not in observed_df.columns:
            raise ValueError(f"'{date_col}' missing in observed data.")
        if outcome_var not in observed_df.columns:
            raise ValueError(f"'{outcome_var}' missing in observed data.")

        obs_tmp = observed_df[[ward_col, date_col, outcome_var]].copy()
        obs_tmp[date_col] = pd.to_datetime(obs_tmp[date_col], errors="coerce")
        obs_tmp = obs_tmp.dropna(subset=[date_col])
        
        if not obs_tmp.empty and obs_tmp[outcome_var].notna().any():
            last_obs_date = obs_tmp.loc[obs_tmp[outcome_var].notna(), date_col].max()
            if pd.notna(last_obs_date):
                obs_last = obs_tmp.loc[obs_tmp[date_col] == last_obs_date, [ward_col, outcome_var]].copy()
                obs_last_map = dict(zip(obs_last[ward_col], obs_last[outcome_var]))

    for h in horizons:
        future_file = os.path.join(results_dir, f"{outcome_var}_pred_{model}_{h}_future_months.csv")
        if not os.path.exists(future_file):
            print(f"⚠️ No future file for h={h}. Skipping.")
            continue

        # Match historical prediction files
        pat_common = fr"{re.escape(outcome_var)}_pred_{re.escape(model)}_{h}_36m_\d+_\d+"
        pattern_with_to = re.compile(pat_common + r"_to_\d+_\d+\.csv$")
        pattern_no_to   = re.compile(pat_common + r"\.csv$")

        all_files = os.listdir(results_dir)
        historic_candidates = [f for f in all_files if (
            pattern_with_to.fullmatch(f) or pattern_no_to.fullmatch(f)
        )]

        # Load future
        df_future = pd.read_csv(future_file)
        if df_future.empty:
            print(f"⚠️ Future file for h={h} is empty. Skipping.")
            continue
        if "time_horizon" not in df_future.columns:
            df_future["time_horizon"] = h
        if ward_col not in df_future.columns or "time_period" not in df_future.columns:
            raise ValueError(f"'{ward_col}' or 'time_period' missing in future file.")
        df_future["time_period"] = pd.to_datetime(df_future["time_period"], errors="coerce")

        # If no historic: just save future-only
        if not historic_candidates:
            df_out = df_future.sort_values([ward_col, "time_period", "time_horizon"]).reset_index(drop=True)
            min_date = df_out["time_period"].min()
            max_date = df_out["time_period"].max()
            new_filename = f"{outcome_var}_pred_{model}_{h}_36m_{min_date.year}_{min_date.month}_to_{max_date.year}_{max_date.month}.csv"
            df_out.to_csv(os.path.join(results_dir, new_filename), index=False)
            print(f"✅ No history for h={h}. Saved future-only file: {new_filename}")
            if delete_inputs:
                try:
                    os.remove(future_file)
                    print(f"🗑️  Deleted: {os.path.basename(future_file)}")
                except FileNotFoundError:
                    pass
            continue

        # Pick most recent historical
        historic_candidates.sort(key=lambda f: os.path.getmtime(os.path.join(results_dir, f)), reverse=True)
        historic_file = os.path.join(results_dir, historic_candidates[0])
        print(f"🔄 Merging: {os.path.basename(historic_file)} + {os.path.basename(future_file)}")

        df_hist = pd.read_csv(historic_file)
        if df_hist.empty:
            # fallback to future-only
            df_out = df_future.sort_values([ward_col, "time_period", "time_horizon"]).reset_index(drop=True)
            min_date = df_out["time_period"].min()
            max_date = df_out["time_period"].max()
            new_filename = f"{outcome_var}_pred_{model}_{h}_36m_{min_date.year}_{min_date.month}_to_{max_date.year}_{max_date.month}.csv"
            df_out.to_csv(os.path.join(results_dir, new_filename), index=False)
            print(f"✅ Hist empty; saved future-only: {new_filename}")
            if delete_inputs:
                try:
                    os.remove(future_file)
                except FileNotFoundError:
                    pass
            continue

        # Ensure cols & types
        if "time_horizon" not in df_hist.columns:
            df_hist["time_horizon"] = h
        if ward_col not in df_hist.columns or "time_period" not in df_hist.columns:
            raise ValueError(f"Historical missing '{ward_col}' or 'time_period'.")
        df_hist["time_period"] = pd.to_datetime(df_hist["time_period"], errors="coerce")

        # ---- BACKFILL STEP: fill observed for the last observed month in HISTORICAL file ----
        # We DO NOT touch yhat/lower/upper in historical; we only fill the outcome column.
        if last_obs_date is not None and obs_last_map is not None:
            is_last_month = df_hist["time_period"] == last_obs_date
            if is_last_month.any():
                # fill outcome_var with observed value by Ward where available
                def _fill_obs(row):
                    if row[ward_col] in obs_last_map:
                        return obs_last_map[row[ward_col]]
                    return row.get(outcome_var, np.nan)

                df_hist.loc[is_last_month, outcome_var] = df_hist.loc[is_last_month].apply(_fill_obs, axis=1)

                # ensure we don't let the FUTURE file overwrite this month:
                df_future = df_future.loc[df_future["time_period"] != last_obs_date].copy()
                print(f"✔️ Backfilled '{outcome_var}' for last observed month {last_obs_date.date()} in historical; "
                      f"removed that month from future to avoid overwrite.")

        # ---- Keep future on overlap (except last observed month which we already removed) ----
        key_cols = [ward_col, "time_period", "time_horizon"]
        future_keys = set(map(tuple, df_future[key_cols].to_records(index=False)))
        overlap_mask = df_hist[key_cols].apply(tuple, axis=1).isin(future_keys)
        n_overlap = int(overlap_mask.sum())
        if n_overlap > 0:
            print(f"🔁 Overlap rows for h={h}: {n_overlap}. Keeping FUTURE rows.")
        df_hist_no_overlap = df_hist.loc[~overlap_mask].copy()

        # Merge & write
        df_merged = pd.concat([df_hist_no_overlap, df_future], ignore_index=True)
        df_merged = df_merged.sort_values([ward_col, "time_period", "time_horizon"]).reset_index(drop=True)

        min_date = df_merged["time_period"].min()
        max_date = df_merged["time_period"].max()
        new_filename = f"{outcome_var}_pred_{model}_{h}_36m_{min_date.year}_{min_date.month}_to_{max_date.year}_{max_date.month}.csv"
        df_merged.to_csv(os.path.join(results_dir, new_filename), index=False)
        print(f"✅ Saved merged file: {new_filename}")

        if delete_inputs:
            for f in (historic_file, future_file):
                try:
                    os.remove(f)
                    print(f"🗑️  Deleted input: {os.path.basename(f)}")
                except FileNotFoundError:
                    pass

#---------------------------------------------------------------------
# Modeling
#---------------------------------------------------------------------
ph = [1, 2, 3]
model_name = ['hb']
# === Run forecasts for two outcomes in one go ===
outcomes = ["wasting_smoothed", "wasting_risk_smoothed"]

df_valid = data.dropna(subset=["wasting_smoothed"])

if "Year" in df_valid.columns and "Month" in df_valid.columns:
    # Sort chronologically
    df_valid = df_valid.sort_values(["Year", "Month"])
    last_year_muac  = int(df_valid["Year"].iloc[-1])
    last_month_muac = int(df_valid["Month"].iloc[-1])

elif "time_period" in df_valid.columns:
    df_valid["time_period"] = pd.to_datetime(df_valid["time_period"])
    last_date = df_valid["time_period"].max()
    last_year_muac, last_month_muac = last_date.year, last_date.month
else:
    raise ValueError("No date columns found (expected 'Year'/'Month' or 'time_period').")

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

data_val = data.copy()  

for outcome_var in outcomes:

    # Use the actual last observed calendar month
    mask = data_val[outcome_var].notna()
    last_obs_date = pd.to_datetime(data_val.loc[mask, "time_period"]).max()

    if pd.isna(last_obs_date):
        raise ValueError(f"No valid time_period for outcome '{outcome_var}'")

    last_obs_year  = int(last_obs_date.year)
    last_obs_month = int(last_obs_date.month)  # no zero pad to match your other files
    fi_path = os.path.join(
        results_dir,
        f"Feature_Importances_{outcome_var}_{last_obs_year}_{last_obs_month}.csv"
    )

    # Create the CSV with headers if it doesn't exist
    if not os.path.exists(fi_path):
        pd.DataFrame(
            columns=["feature", "importance", "model", "time_horizon", "time_period"]
        ).to_csv(fi_path, index=False)


    # 2) Iterate models and horizons
    for m in model_name:
        for horizon in ph:
            predictions_df = pd.DataFrame()

            # Feature list for this model & horizon (keeps your globals() pattern)
            feat_list_name = f"{m}_m{horizon}"
            if feat_list_name not in globals():
                print(f"⚠️  Missing feature list: {feat_list_name}. Skipping {m} h={horizon}.")
                continue
            feat_list = [c for c in globals()[feat_list_name] if c in data.columns]

            last_obs_code = data_val.loc[mask, "time_cont_enc"].max()

            # Train/predict split at last_obs_month + horizon
            train, predict = train_test_split(data, data_val, last_obs_code, horizon)

            # Clean training rows (ensure outcome exists)
            train_clean = train[train[outcome_var].notna()].copy()
            if train_clean.empty:
                print(f"⚠️  Empty training set for {outcome_var}, model={m}, h={horizon}. Skipping.")
                continue

            # Build matrices (only features present in each split)
            trainX = train_clean[feat_list]
            trainY = train_clean[outcome_var]
            testX  = predict[feat_list]
            testY  = predict[outcome_var]

            if testX.empty:
                print(f"⚠️  No test rows for {outcome_var}, model={m}, h={horizon}. Skipping.")
                continue

            # 3) Model
            model = XGBRegressor(
                n_estimators=1000, max_depth=7, eta=0.1,
                subsample=0.9, colsample_bylevel=1, seed=12345
            )
            model.fit(trainX, trainY)

            # 4) Feature importance (append)
            fi = pd.DataFrame({
                "feature": trainX.columns,
                "importance": model.feature_importances_,
                "model": m,
                "time_horizon": horizon,
                "time_period": predict["time_period"].iloc[0]
            })
            fi.to_csv(fi_path, mode="a", header=False, index=False)

            # 5) Prediction + bootstrap CIs
            yhat = model.predict(testX)
            num_iterations = 1000
            all_predictions = bootstrap_iteration(
                model, train_clean, testX, feat_list, num_iterations
            )
            lower_bound = np.percentile(all_predictions, 5,  axis=0)
            upper_bound = np.percentile(all_predictions, 95, axis=0)

            df_yhat = pd.DataFrame(
                data={
                    "yhat": np.maximum(yhat, 0),
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                },
                index=testX.index.copy()
            )

            df_out = df_yhat.join(predict[["Ward", "time_cont_enc", 
                                           "time_period"]])
            
            df_out["train_size"] = len(trainX)
            df_out["test_size"]  = len(testX)
            df_out["time_horizon"] = horizon  # helpful when merging later

            predictions_df = pd.concat([predictions_df, df_out], ignore_index=True)

            # 6) Save predictions for this outcome/model/horizon
            pred_file = os.path.join(results_dir, f"{outcome_var}_pred_{m}_{horizon}_future_months.csv")
            predictions_df.to_csv(pred_file, index=False)
            globals()[f"predictions_{horizon}"] = predictions_df

        # 7) Merge with historical per outcome+model (keeps future on overlap)
        merge_and_replace_predictions(
                results_dir=results_dir,
                model=m,
                outcome_var=outcome_var,
                horizons=(1, 2, 3),
                delete_inputs=True,
                observed_df = data,   
                observed_file=  None,          
                ward_col= "Ward",
                date_col= "time_period",
            )




#=-----------------
# FEATURE IMPORTANCE PLOTS
#-------------------------------------
# Renaming the features
TOKEN_MAP = {
    # population / access
    "travel_time_to_cities_2015": "Travel time to cities (2015)",
    "density_2015":               "Population density (2015)",
    "density_2020":               "Population density (2020)",
    "delta_2015_2020":            "Population density change (2015→2020)",

    # temperature & derived
    "avg_temp_z_score":           "Average temperature z-score",
    "hot_days_z_score":           "Hot days z-score",
    "cold_days_z_score":          "Cold days z-score",
    "consec_hot_days_z_score":    "Consecutive hot days z-score",
    "consec_cold_days_z_score":   "Consecutive cold days z-score",

    # precipitation & wet/dry spells
    "precip_total_z_score":       "Total precipitation z-score",
    "wet_days_z_score":           "Wet days z-score",
    "dry_days_z_score":           "Dry days z-score",
    "consec_wet_days_z_score":    "Consecutive wet days z-score",
    "consec_dry_days_z_score":    "Consecutive dry days z-score",

    # conflict & fatalities
    "conflict":                   "Conflict events",
    "fatalities":                 "Conflict fatalities",

    # vegetation indices (overall)
    "NDVI_mean":                  "NDVI mean",
    "EVI_mean":                   "EVI mean",
    "NDVI_z_score":               "NDVI z-score",
    "EVI_z_score":                "EVI z-score",

    # prior outcome features
    "wasting":                    "Wasting prevalence",
    "wasting_risk":               "Wasting risk",
    "wasting_delta":              "Wasting prevalence change",
    "wasting_risk_delta":         "Wasting risk change",
    "wasting_sm":                 "Wasting prevalence (smoothed)",
    "wasting_sm_risk":            "Wasting risk (smoothed)",

    # land cover shares
    "bare":                       "Bare ground share",
    "built":                      "Built-up area share",
    "crops":                      "Cropland share",
    "flooded_vegetation":         "Flooded vegetation share",
    "grass":                      "Grassland share",
    "shrub_and_scrub":            "Shrub & scrub share",
    "trees":                      "Tree cover share",
    "water":                      "Surface water share",
}

# Land-cover class labels used when the feature name embeds the class
LC_LABEL = {
    "crops": "cropland",
    "grass": "grassland",
    "shrub_and_scrub": "shrub & scrub",
    "trees": "tree cover",
}

# helper: format pieces (lag, window, radius, deltas)
def _fmt_suffix(lag=None, window_12m=False, radius_km=None):
    parts = []
    if window_12m:
        parts.append("12-month window")
    if radius_km:
        parts.append(f"within {radius_km} km")
    if lag:
        parts.append(f"lag {lag} mo")
    return " (" + ", ".join(parts) + ")" if parts else ""

def pretty_feature(name: str) -> str:
    n = str(name)

    # --- capture lag first (but don't mutate n yet) ---
    lag = None
    m = re.search(r"_lag_(\d+)$", n)
    if m:
        lag = int(m.group(1))
        n_wo_lag = n[:m.start()]
    else:
        n_wo_lag = n

    # --- detect delta on the lagless name ---
    is_delta = n_wo_lag.endswith("_delta")
    base_key = n_wo_lag[:-6] if is_delta else n_wo_lag  # strip "_delta" if present

    # window + radius extraction (work on base_key)
    window_12m = "_12m_" in base_key
    base_key = base_key.replace("_12m_", "_")
    radius_km = None
    m = re.search(r"_(\d{2,3})km_", base_key)
    if m:
        radius_km = int(m.group(1))
        base_key = base_key.replace(m.group(0), "_")

    # --- make the base label (without delta wording) ---
    # Try TOKEN_MAP first (for things like wasting, wasting_risk, etc.)
    if base_key in TOKEN_MAP:
        base_label = TOKEN_MAP[base_key]
    else:
        # vegetation by land cover (e.g., NDVI_z_score_crops)
        m = re.match(r"(NDVI|EVI)_(mean|z_score)_(\w+)$", base_key)
        if m:
            idx, kind, clazz = m.groups()
            lc = LC_LABEL.get(clazz, clazz.replace("_", " "))
            base_label = f"{idx} {'mean' if kind=='mean' else 'z-score'} — {lc}"
        # land-cover share base keys
        elif base_key in {"bare","built","crops","flooded_vegetation","grass","shrub_and_scrub","trees","water"}:
            base_label = TOKEN_MAP[base_key]  # e.g., "Cropland share"
        else:
            # fallback: title-case words, keep acronyms
            base_label = " ".join(w.upper() if w in {"NDVI","EVI"} else w.capitalize()
                                  for w in base_key.split("_"))

    # --- if this feature is a delta, say so explicitly on the base label ---
    if is_delta:
        # If label ends with "share", turn it into "share change"
        if base_label.endswith("share"):
            base_label = base_label.replace("share", "share change")
        # If it already says "prevalence" or similar, append "change"
        elif not base_label.lower().endswith("change"):
            base_label = f"{base_label} change"

    # --- suffix for lag/window/radius (no generic Δ anymore) ---
    return base_label + _fmt_suffix(lag=lag, window_12m=window_12m, radius_km=radius_km)


# =========================
# Config
# =========================
DATA_DIR = "results"       # where your Feature_Importances_*.csv live
OUT_DIR  = "figures"
OUTCOMES = ["wasting_smoothed", "wasting_risk_smoothed"]
os.makedirs(OUT_DIR, exist_ok=True)

def relabel_features(df: pd.DataFrame) -> pd.DataFrame:
    if "feature_renamed" in df.columns:
        return df
    out = df.copy()
    out["feature_renamed"] = out["feature"]
    return out

# =========================
# Filename parsing + latest finder
# =========================
FNAME_RE = re.compile(
    r"Feature_Importances_(?P<outcome>.+?)_(?P<year>\d{4})_(?P<month>\d{1,2})\.csv$",
    re.IGNORECASE
)

def find_latest_file(outcome: str):
    pattern = os.path.join(DATA_DIR, f"Feature_Importances_{outcome}_*.csv")
    best = None  # (year, month, mtime, path)
    for p in glob.glob(pattern):
        m = FNAME_RE.search(os.path.basename(p))
        if not m: 
            continue
        if m.group("outcome") != outcome:
            continue
        y = int(m.group("year")); mth = int(m.group("month"))
        key = (y, mth, os.path.getmtime(p))
        if (best is None) or (key > best[:3]):
            best = (y, mth, os.path.getmtime(p), p)
    if not best:
        raise FileNotFoundError(f"No file found for outcome '{outcome}' under {DATA_DIR}")
    y, mth, _, path = best
    as_of = pd.Timestamp(y, mth, 1)
    return path, as_of

# =========================
# Pooling helpers (collapse all lags; bucket wasting/risk)
# =========================
WASTING_GROUP_LABEL      = "Wasting (time-series signal)"
WASTING_RISK_GROUP_LABEL = "Wasting risk (time-series signal)"

def base_no_lag(name: str) -> str:
    return re.sub(r"_lag_\d+$", "", str(name))

def is_wasting_base(base: str) -> bool:
    b = base.lower()
    return b.startswith(("wasting", "wasting_sm", "wasting_delta"))

def is_wasting_risk_base(base: str) -> bool:
    b = base.lower()
    return ("risk" in b) and b.startswith(("wasting", "wasting_sm", "wasting_delta"))

def pretty_feature_minimal(name: str) -> str:
    parts = []
    for w in str(name).split("_"):
        parts.append(w.upper() if w in {"NDVI","EVI"} else (w.capitalize() if w else w))
    return " ".join(parts)

def pooled_label_from_row(row) -> str:
    raw = row.get("feature", row.get("feature_renamed", ""))
    base = re.sub(r"_lag_\d+$", "", str(raw)).strip()

    if is_wasting_risk_base(base):
        return "Wasting risk (time-series signal)"
    if is_wasting_base(base):
        return "Wasting (time-series signal)"

    return pretty_feature(base)


# =========================
# ---------- multi-horizon top-5 (labels on LEFT y-axis) ----------
POOL_LAGS = True   # pool lagged versions into families
TOPK = 5

# ---------- HELPERS ----------
def _read_latest(outcome: str):
    path, as_of = find_latest_file(outcome)
    df = pd.read_csv(path)
    df = relabel_features(df)
    if "feature_renamed" not in df.columns or df["feature_renamed"].isna().all():
        df["feature_renamed"] = df["feature"].apply(pretty_feature)
    df["time_period"] = pd.to_datetime(df.get("time_period", as_of), errors="coerce")
    return df, as_of, path

def _label_col(row):
    return pooled_label_from_row(row) if POOL_LAGS else row.get("feature_renamed", row.get("feature", ""))

import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ---------- PLOT ----------
for outcome_var in OUTCOMES:
    df, _, src = _read_latest(outcome_var)

    horizons, top_by_h, order_in_h = [], {}, {}
    for h in (1, 2, 3):
        dh = df[df["time_horizon"] == h].copy()
        if dh.empty:
            continue
        dh["group"] = dh.apply(_label_col, axis=1)
        agg = (dh.groupby("group", as_index=False)["importance"]
                 .sum()
                 .sort_values("importance", ascending=False))
        top = agg.head(TOPK).copy()
        if len(top):
            horizons.append(h)
            top_by_h[h] = top
            order_in_h[h] = list(top["group"])

    if not horizons:
        print(f"[{outcome_var}] No horizons 1–3 found in {src}")
        continue

    # y-order: H1 top-5, then H2-only additions, then H3-only additions
    row_order, seen = [], set()
    for h in (1, 2, 3):
        if h in order_in_h:
            for g in order_in_h[h]:
                if g not in seen:
                    row_order.append(g); seen.add(g)

    # stable colors across panels
    palette = sns.color_palette("pastel", n_colors=len(row_order))
    COLOR = dict(zip(row_order, palette))

    # panel titles like "1-month prediction — YYYY-MM"
    titles = {}
    for h in horizons:
        m = pd.to_datetime(df.loc[df["time_horizon"] == h, "time_period"].max(), errors="coerce")
        titles[h] = f"{h}-month prediction — {m:%Y-%m}" if pd.notna(m) else f"{h}-month prediction"

    # shared y positions
    ypos = np.arange(len(row_order))

    n = len(horizons)
    fig, axes = plt.subplots(
        1, n, sharey=True,
        figsize=(14.5, max(4.8, 0.54*len(row_order))),
        constrained_layout=False
    )
    if n == 1:
        axes = [axes]

    for i, (ax, h) in enumerate(zip(axes, horizons)):
        d = top_by_h[h].set_index("group")
        vals = [float(d["importance"].get(g, 0.0)) for g in row_order]
        cols = [COLOR[g] for g in row_order]
        bars = ax.barh(ypos, vals, color=cols, edgecolor="none")

        # aesthetics: subtle spines + x & y gridlines
        for side, spine in ax.spines.items():
            spine.set_visible(side in {"left", "bottom"})
            spine.set_linewidth(0.8)
            spine.set_alpha(0.25)
        ax.set_axisbelow(True)
        ax.grid(axis="x", linestyle=":", alpha=0.55)     # vertical dotted lines
        ax.grid(axis="y", linestyle="-", alpha=0.12)     # faint horizontal guides
        ax.set_facecolor("white")

        # axes & titles
        ax.set_title(titles[h], fontsize=12.2, pad=10)
        ax.set_xlabel("Importance")
        ax.set_ylim(-0.5, len(row_order)-0.5)
        ax.invert_yaxis()

        if i == 0:
            ax.set_yticks(ypos)
            ax.set_yticklabels(row_order, fontsize=10)
            ax.set_ylabel("Feature")
        else:
            ax.set_yticks(ypos)
            ax.tick_params(axis="y", labelleft=False)

        # number labels at bar ends
        xmax = max(vals) if vals else 0.0
        pad = 0.02 * max(0.1, xmax)
        for rect, v in zip(bars, vals):
            if v > 0:
                ax.text(rect.get_width() + pad,
                        rect.get_y() + rect.get_height()/2,
                        f"{v:.2f}", va="center", fontsize=8)

    # more room for long y labels & titles (avoid overlap)
    fig.subplots_adjust(left=0.40, right=0.98, bottom=0.12, top=0.88, wspace=0.22)

    # save
    stamp2 = pd.to_datetime(df["time_period"].max()).strftime("%Y-%m")
    out_png = f"figures/top_features_{outcome_var}_{stamp2}_multiH.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[{outcome_var}] Saved:", out_png)

    #Save last three images, then eliminate oldest ones
    K = 3
    pattern = os.path.join(OUT_DIR, f"top_features_{outcome_var}_*_multiH.png")
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    for old in candidates[K:]:
        try:
            os.remove(old)
        except OSError as e:
            print(f"Could not delete {old}: {e}")

#    plt.show()
#    plt.close(fig)

#---------------------------
# Eliminating old csv file with feat imp
#---------------------------
    from datetime import datetime

    DATA_DIR = "results"  # adjust if needed

    CSV_RE = re.compile(
        r"Feature_Importances_(?P<outcome>.+?)_(?P<ym>\d{4}_\d{1,2})\.csv$",
        re.IGNORECASE
    )

    def _parse_ym(ym: str) -> datetime | None:
        try:
            return datetime.strptime(ym, "%Y_%m")
        except Exception:
            return None

    def _rank_csvs_by_stamp(paths):
        """Return list sorted oldest->newest by YYYY_MM in filename; fallback to mtime."""
        def key(p: Path):
            m = CSV_RE.search(p.name)
            if m and m.group("ym"):
                dt = _parse_ym(m.group("ym"))
                if dt is not None:
                    return (dt, p.stat().st_mtime)
            return (datetime.fromtimestamp(p.stat().st_mtime), p.stat().st_mtime)
        return sorted(paths, key=key)

    def prune_old_csvs(outcome: str, keep: int = 1, dry_run: bool = False):
        """
        Keep only the newest `keep` CSVs for an outcome; delete the rest.
        Selection is by YYYY_MM in filename; mtime is a tiebreaker.
        """
        pattern = os.path.join(DATA_DIR, f"Feature_Importances_{outcome}_*.csv")
        paths = [Path(p) for p in glob.glob(pattern)]
        if not paths or keep <= 0:
            return

        ranked = _rank_csvs_by_stamp(paths)  # oldest -> newest
        to_delete = ranked[:-keep] if len(ranked) > keep else []
        for p in to_delete:
            if dry_run:
                print(f"[dry-run] Would delete: {p}")
            else:
                try:
                    p.unlink()
                    print(f"[csv] Deleted old: {p}")
                except OSError as e:
                    print(f"[csv] Could not delete {p}: {e}")

    prune_old_csvs(outcome_var, keep=1)


