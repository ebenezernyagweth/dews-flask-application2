import pandas as pd
import geopandas as gpd
import os 
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import BoundaryNorm
import calendar
import matplotlib.dates as mdates
import gc
import glob
from scipy.stats import gamma as gamma_dist, norm

#==============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
INPUT = os.path.join(PARENT_DIR, "intermediary_datasets")
SHAPE = os.path.join(PARENT_DIR, "shapefiles")

# Change directory to the general folder that contains intermediary_datasets folder
os.chdir(PARENT_DIR)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#==============================================================
# Standardised indices for precipitation
#--------------------------------------------------------------
# A per-ward z-score fails for precipitation in arid ward-months: the baseline
# is (near-)always dry, so sigma is zero or tiny and the z-score returns inf or
# absurd values. Not a sample-size problem - those wards have 17 baseline years
# - but monthly rainfall in the ASAL is zero-inflated and a z-score assumes a
# distribution with real spread.
#
#   precip_total  -> SPI (gamma fitted to the wet years, explicit mass at zero)
#   count vars    -> empirical normal score (nonparametric standardised index,
#                    Farahmand & AghaKouchak 2015)
#
# Both bounded at +/-3.09 (WMO convention, p = 0.001).
#==============================================================
SPI_CLIP = 3.09
MIN_BASELINE_N = 10      # minimum baseline years per ward-month cell
MIN_NONZERO = 6          # minimum wet years needed to fit a gamma
EPS = 1e-6


def _thom_gamma(x):
    """Thom's ML estimator for gamma shape/scale; None if not defensible."""
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    if len(x) < MIN_NONZERO:
        return None
    xbar = x.mean()
    A = np.log(xbar) - np.log(x).mean()
    if not np.isfinite(A) or A <= 0:
        return None
    alpha = (1.0 + np.sqrt(1.0 + 4.0 * A / 3.0)) / (4.0 * A)
    beta = xbar / alpha
    if not (np.isfinite(alpha) and np.isfinite(beta)) or alpha <= 0 or beta <= 0:
        return None
    return alpha, beta


def _normal_score(baseline, target):
    """
    Rank of each target value within its ward-month baseline, mapped through the
    normal quantile function. Midranks handle ties, so repeated zeros (or
    repeated 31s) land together rather than spreading across the scale.
    Bounded by baseline length: with 17 years the extreme is about +/-1.9.
    """
    b = np.sort(np.asarray(baseline, dtype=float))
    n = len(b)
    le = np.searchsorted(b, target, side="right")
    lt = np.searchsorted(b, target, side="left")
    p = ((le + lt) / 2.0 + 0.5) / (n + 1.0)
    return np.clip(norm.ppf(np.clip(p, EPS, 1 - EPS)), -SPI_CLIP, SPI_CLIP)


def _spi_cell(baseline, target):
    """SPI for one ward-month cell. Returns (values, method)."""
    b = np.asarray(baseline, dtype=float)
    b = b[np.isfinite(b)]
    t = np.asarray(target, dtype=float)
    out = np.full(t.shape, np.nan)

    if len(b) < MIN_BASELINE_N:
        return out, "insufficient_baseline"

    q = float((b == 0).sum()) / len(b)

    if q >= 1.0:
        # Baseline never records rain here: a dry month is unremarkable, any
        # rain at all is unprecedented. A modelling assumption, not a computed
        # value - flagged via the method column.
        out = np.where(t > 0, SPI_CLIP, 0.0)
        out[~np.isfinite(t)] = np.nan
        return out, "all_dry_baseline"

    params = _thom_gamma(b)
    if params is None:
        vals = _normal_score(b, t)
        vals[~np.isfinite(t)] = np.nan
        return vals, "empirical_fallback"

    alpha, beta = params
    with np.errstate(invalid="ignore"):
        G = gamma_dist.cdf(t, a=alpha, scale=beta)
    H = q + (1.0 - q) * G
    vals = np.clip(norm.ppf(np.clip(H, EPS, 1 - EPS)), -SPI_CLIP, SPI_CLIP)
    vals[~np.isfinite(t)] = np.nan
    return vals, "gamma"


#==============================================================
# Get previous and updated MUAC data
#-------------------------------------
muac_data = pd.read_pickle(
    os.path.join(INPUT,
                 'Kenya_NDMA_MUAC_ward_level_2021_01_Onwards.pkl'))
new_muac_data = pd.read_pickle(
    os.path.join(INPUT,
                 'Kenya_NDMA_MUAC_23_counties.pkl'))
min_num_obs_ward = 35

# FIX 1: 'Year'.min() and 'month_num'.min() were taken INDEPENDENTLY. If
# new_muac_data spans a year boundary (e.g. Nov 2025 - Jun 2026) the month min
# comes from the 2026 rows, producing "2025_01" - a label matching no file that
# script 2 ever wrote. Derive both ends from the same period instead.
_new_periods = pd.PeriodIndex(
    pd.to_datetime(dict(year=new_muac_data['Year'],
                        month=new_muac_data['month_num'],
                        day=1)),
    freq='M')
_start_period, _end_period = _new_periods.min(), _new_periods.max()

start_Month_Year = f"{_start_period.year}_{_start_period.month:02d}"
end_Month_Year = f"{_end_period.year}_{_end_period.month:02d}"

start_Month_Year1 = f"{calendar.month_abbr[_start_period.month]}_{_start_period.year}"
end_Month_Year1 = f"{calendar.month_abbr[_end_period.month]}_{_end_period.year}"

# FIX 2: pre-flight report. Script 2 names its outputs from its own
# --start_date/--end_date; script 3 reconstructs the names from new_muac_data.
# Nothing enforces that they agree, and a mismatch silently falls back to old
# data. Report the situation up front instead of discovering it 40 minutes in.
_expected_new_files = {
    "ERA5 temperature":     f"era5_stats_{start_Month_Year}_to_{end_Month_Year}.pkl",
    "CHIRPS precipitation": f"chirps_stats_{start_Month_Year}_to_{end_Month_Year}.pkl",
    "MODIS NDVI/EVI":       f"modis_ndvi_evi_stats_{start_Month_Year}_to_{end_Month_Year}.pkl",
    "MODIS EVI by land use": f"modis_evi_by_land_use_stats_{start_Month_Year}_to_{end_Month_Year}.pkl",
    "Land use percentage":  f"land_use_pct_stats_{start_Month_Year}_to_{end_Month_Year}.pkl",
    "ACLED conflict 100km": f"ACLED_conflict_12m_running_sum_by_polygon_100kmcutoff_{end_Month_Year1}.pkl",
    "ACLED conflict 500km": f"ACLED_conflict_12m_running_sum_by_polygon_500kmcutoff_{end_Month_Year1}.pkl",
}
print("\n" + "=" * 70)
print(f"New MUAC data covers {_start_period} to {_end_period}")
print("Expected new source files in", INPUT)
print("=" * 70)
_absent = []
for _label, _fname in _expected_new_files.items():
    _found = os.path.exists(os.path.join(INPUT, _fname))
    print(f"  [{'OK ' if _found else 'MISSING'}] {_label:24s} {_fname}")
    if not _found:
        _absent.append(_label)
if _absent:
    print("\n  !! The sources above will silently fall back to previously merged")
    print("     data. Candidate new-export files present in the folder:")
    for _cand in sorted(glob.glob(os.path.join(INPUT, "*_stats_*.pkl"))):
        _base = os.path.basename(_cand)
        if "_Onwards" not in _base:
            print(f"       {_base}")
    print("     If one of these is the file you meant, rename it to the expected")
    print("     name above, or rerun script 2 with dates matching the MUAC range.")
print("=" * 70 + "\n")

counties = gpd.read_file(os.path.join(SHAPE, 
                                "ken_admbnda_adm1_iebc_20191031.shp"))
county_names = [
    'Baringo', 'Embu', 'Garissa', 'Isiolo', 'Kajiado', 'Kilifi', 'Kitui', 'Kwale', 
    'Laikipia', 'Lamu', 'Makueni', 'Mandera', 'Marsabit', 'Meru', 'Narok', 'Nyeri', 
    'Samburu', 'Taita Taveta', 'Tana River', 'Tharaka-Nithi', 'Turkana', 'Wajir', 'West Pokot'
]

valid_counties = counties[counties['ADM1_EN'].isin(county_names)]
wards_shapefile = gpd.read_file(os.path.join(SHAPE, 'Kenya_wards_NDMA.shp'))

#==============================================================
class WastingPrevalenceDatasetBuilder:
    def __init__(self, input_path, wards_shapefile, 
                 valid_counties, historic_muac_data, new_muac_data, polygon_id_col):
        self.input_path = input_path
        self.muac_data = historic_muac_data
        self.new_muac_data = new_muac_data
        self.merged_data = None
        self.wards_shapefile = wards_shapefile
        self.valid_counties = valid_counties
        self.polygon_id_col = polygon_id_col

        print("Initialized Dataset Builder.")

    def _max_consecutive_gap(self, missing_idx):
        """
        Given a DatetimeIndex of missing months (freq='MS'),
        return the length of the longest run of consecutive missing months.
        """
        if missing_idx.empty:
            return 0

        months = missing_idx.to_period('M').sort_values()

        max_run = 1
        current_run = 1

        for prev, curr in zip(months[:-1], months[1:]):
            if curr - prev == 1:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
            else:
                current_run = 1

        return max_run        

    def join_datasets_and_eliminate_duplicates(self):
        print("Joining MUAC datasets and eliminating duplicates...")

        self.muac_data = self.muac_data.rename(columns={'month_num': 'month', 'Year': 'year'})
        self.new_muac_data = self.new_muac_data.rename(columns={'month_num': 'month', 'Year': 'year'})

        self.muac_data = pd.concat([self.muac_data, self.new_muac_data], ignore_index=True)

        dupes = self.muac_data[self.muac_data.duplicated(subset=['Ward', 'year', 'month'], keep=False)]
        if not dupes.empty:
            print("Found duplicates:")
            print(dupes.sort_values(['Ward', 'year', 'month'])[['Ward', 'year', 'month', 'wasting']])
        else:
            print("No duplicates found.")

        self.muac_data = self.muac_data.drop_duplicates(subset=['Ward', 'year', 'month'], keep='last')

        print(self.muac_data.head())

        year_month_counts = (
            self.muac_data
            .groupby(['year', 'month'])
            .size()
            .reset_index(name='num_obs')
            .sort_values(['year', 'month'])
        )

        for _, row in year_month_counts.iterrows():
            print(f"Year: {row['year']}, Month: {row['month']}, Observations: {row['num_obs']}")


    def eliminate_prevalences_with_insufficient_obs(self, min_num_obs_ward=35, plot=False):
        """
        Filters out wards with fewer than the minimum number of observations per month.
        """
        self.muac_data_filtered = self.muac_data[self.muac_data['obs_per_month'] >= min_num_obs_ward]
        
        print(f"Original number of wards: {self.muac_data.Ward.nunique()}, "
            f"Wards left after filtering: {self.muac_data_filtered.Ward.nunique()}")

        missing_wards = self.muac_data[~self.muac_data['Ward'].isin(
            self.muac_data_filtered['Ward'])].drop_duplicates(subset=['Ward'])

        self.wards_NDMA = self.wards_shapefile[
            self.wards_shapefile['Ward'].isin(self.muac_data['Ward'])][['Ward', 'geometry']]
        self.wards_NDMA_filtered = self.wards_shapefile[
            self.wards_shapefile['Ward'].isin(
            self.muac_data_filtered['Ward'])][['Ward', 'geometry']]
        missing_wards_shape = self.wards_shapefile[self.wards_shapefile['Ward'].isin(
            missing_wards['Ward'])][['Ward', 'geometry']]

        if plot:
            print("Plotting original and eliminated wards after minimum observations filtering...")
            fig, ax = plt.subplots(figsize=(10, 8))
            missing_wards_shape.plot(ax=ax, color='red', edgecolor='black')
            self.wards_NDMA_filtered.plot(ax=ax, color='lightblue', edgecolor='black')
            missing_patch = mpatches.Patch(color='red', label='Eliminated Wards')
            filtered_patch = mpatches.Patch(color='lightblue', label='Remaining Wards')
            plt.legend(handles=[missing_patch, filtered_patch], loc='lower left', fontsize='x-small')
            plt.title("Wards After Minimum Observations Filtering")
            plt.tight_layout()

    
    def clean_for_data_continuity(self, total_months_min=6, max_gap_size=3, 
                                    folder_images="plots", save_plots=True, plot=False):
        """
        Cleans MUAC data for continuity.
        """
        print("Starting data continuity cleaning...")

        # Fix SettingWithCopyWarning — make a proper copy first
        self.muac_data_filtered = self.muac_data_filtered.copy()
        self.muac_data_filtered['year'] = self.muac_data_filtered['year'].astype(int)
        self.muac_data_filtered['month'] = self.muac_data_filtered['month'].astype(int)

        self.muac_data_filtered['date'] = pd.to_datetime(
            self.muac_data_filtered[['year', 'month']].assign(day=1),
            errors='raise'
        )

        ward_summary = (
            self.muac_data_filtered.groupby('Ward')
            .agg(
                first_observation=('date', 'min'),
                last_observation=('date', 'max'),
                total_months=('date', 'nunique')
            )
            .reset_index()
        )

        # Pre-group to avoid full DataFrame scan per ward inside apply
        grouped = self.muac_data_filtered.dropna(subset=['wasting']).groupby('Ward')

        def calculate_gaps(row):
            ward_name = row['Ward']
            if ward_name not in grouped.groups:
                return pd.Series({
                    'gap_size': 0, 'initial_gap_size': 0, 'end_gap_size': 0,
                    'pre_2024_gap_size': 0, 'post_2024_continuous': False,
                    'continuity_status': 'No data'
                })

            ward_data = grouped.get_group(ward_name).sort_values(['year', 'month'])
            observed_dates = pd.DatetimeIndex(
                pd.to_datetime(ward_data[['year', 'month']].assign(day=1))
            ).sort_values().unique()

            dataset_start = self.muac_data_filtered['date'].min()
            dataset_end = self.muac_data_filtered['date'].max()

            full_range = pd.date_range(start=row['first_observation'],
                                       end=row['last_observation'], freq='MS')
            missing_months = full_range.difference(observed_dates)
            gap_size = self._max_consecutive_gap(missing_months)

            initial_gap_range = pd.date_range(
                start=dataset_start,
                end=row['first_observation'] - pd.DateOffset(months=1),
                freq='MS'
            )
            initial_gap_size = len(initial_gap_range.difference(observed_dates))

            end_gap_range = pd.date_range(
                start=row['last_observation'] + pd.DateOffset(months=1),
                end=dataset_end, freq='MS'
            )
            end_gap_size = len(end_gap_range.difference(observed_dates))

            jan_2024 = pd.Timestamp('2024-01-01')
            pre_2024_range = pd.date_range(
                start=row['first_observation'],
                end=jan_2024 - pd.DateOffset(months=1),
                freq='MS'
            )
            pre_2024_gap_size = len(pre_2024_range.difference(observed_dates))

            post_2024_time = observed_dates[observed_dates >= jan_2024]
            if post_2024_time.size == 0:
                post_2024_continuous = False
            else:
                post_2024_range = pd.date_range(
                    start=jan_2024, end=row['last_observation'], freq='MS')
                post_2024_continuous = (len(post_2024_range.difference(post_2024_time)) == 0)

            if post_2024_continuous:
                continuity_status = 'Continuous after January 2024'
            elif initial_gap_size > 0 and gap_size == 0:
                continuity_status = 'Continuous after introduction'
            elif gap_size == 0:
                continuity_status = 'Continuous'
            else:
                continuity_status = 'Has gaps'

            return pd.Series({
                'gap_size': gap_size,
                'initial_gap_size': initial_gap_size,
                'end_gap_size': end_gap_size,
                'pre_2024_gap_size': pre_2024_gap_size,
                'post_2024_continuous': post_2024_continuous,
                'continuity_status': continuity_status
            })

        gap_details = ward_summary.apply(calculate_gaps, axis=1)
        final_ward_summary = pd.concat([ward_summary, gap_details], axis=1)
        
        print(final_ward_summary[['Ward', 'first_observation', 'last_observation',
                                  'total_months', 'gap_size', 'initial_gap_size',
                                  'end_gap_size', 'pre_2024_gap_size',
                                  'post_2024_continuous', 'continuity_status']])
        
        valid_wards = final_ward_summary[
            (final_ward_summary['total_months'] >= total_months_min) &
            (final_ward_summary['gap_size'] <= max_gap_size)
        ]
        
        post_2024_only_wards = final_ward_summary[
            (~final_ward_summary['Ward'].isin(valid_wards['Ward'])) &
            (final_ward_summary['post_2024_continuous'])
        ]
        
        valid_wards_list = valid_wards['Ward'].tolist()
        post_2024_only_list = post_2024_only_wards['Ward'].tolist()
        
        self.muac_data_filtered = self.muac_data_filtered[
            (self.muac_data_filtered['Ward'].isin(valid_wards_list)) |
            ((self.muac_data_filtered['Ward'].isin(post_2024_only_list)) &
             (self.muac_data_filtered['year'] >= 2024))
        ].copy()
        
        self.muac_data_filtered['date'] = pd.to_datetime(
            self.muac_data_filtered[['year', 'month']].assign(day=1),
            errors='raise'
        )
        
        start_date = self.muac_data_filtered['date'].min().strftime('%Y_%m')
        end_date   = self.muac_data_filtered['date'].max().strftime('%Y_%m')
        print(f"MUAC data time frame: {start_date} to {end_date}")
        
        filename  = f"Kenya_NDMA_MUAC_ward_level_{start_date}_to_{end_date}.pkl"
        filename2 = f"Kenya_NDMA_MUAC_ward_level_{start_date}_Onwards.pkl"
        
        output_path1 = os.path.join(self.input_path, filename)
        output_path2 = os.path.join(self.input_path, filename2)
        
        self.muac_data_filtered.to_pickle(output_path1)
        self.muac_data_filtered.to_pickle(output_path2)
        
        print(f"Filtered MUAC dataset saved to {output_path2}")
        print(f"Total selected wards: {len(valid_wards_list) + len(post_2024_only_list)}")
        print(f"Wards selected due to post-2024 continuity: {len(post_2024_only_list)}")
        print(f"Final dataset shape: {self.muac_data_filtered.shape}")
        
        # Free gap computation intermediates
        del gap_details, final_ward_summary, ward_summary, valid_wards, post_2024_only_wards
        gc.collect()

        inconsistent_folder = os.path.join(folder_images, "inconsistent_wards")
        summary_folder = os.path.join(folder_images, "summary_plots")
        if save_plots:
            os.makedirs(inconsistent_folder, exist_ok=True)
            os.makedirs(summary_folder, exist_ok=True)
        
        for ward in post_2024_only_list:
            ward_data = self.muac_data[self.muac_data['Ward'] == ward].copy()
            ward_data['time'] = pd.to_datetime(ward_data[['year', 'month']].assign(day=1), errors='coerce')
            ward_data = ward_data.dropna(subset=['time'])
        
            if not ward_data.empty:
                fig, ax1 = plt.subplots(figsize=(12, 6))
                ax1.plot(ward_data['time'], ward_data['wasting'],
                         marker='o', linestyle='-', color='b', label='Wasting Prevalence')
                ax2 = ax1.twinx()
                ax2.plot(ward_data['time'], ward_data['wasting_risk'],
                         marker='x', linestyle='-', color='r', label='Wasting Risk')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Wasting Prevalence', color='b')
                ax2.set_ylabel('Wasting Risk', color='r')
                plt.title(f'Wasting Prevalence Over Time - {ward}')
                plt.tight_layout()
        
                safe_filename = ward.replace(" ", "_").replace("/", "_").replace("\\", "_")
                file_path = os.path.join(inconsistent_folder, f"wasting_prevalence_{safe_filename}.png")
                if save_plots:
                    plt.savefig(file_path)
                    print(f"Saved: {file_path}")
                if plot:
                    plt.show()
                plt.close()
        
        avg_all = self.muac_data.groupby(['year', 'month'])['wasting'].mean().reset_index()
        avg_all['date'] = pd.to_datetime(avg_all[['year', 'month']].assign(day=1), errors='coerce')
        avg_all = avg_all.dropna(subset=['date'])
        
        avg_filtered = self.muac_data_filtered.groupby(['year', 'month'])['wasting'].mean().reset_index()
        avg_filtered['date'] = pd.to_datetime(avg_filtered[['year', 'month']].assign(day=1), errors='coerce')
        avg_filtered = avg_filtered.dropna(subset=['date'])
        
        plt.figure(figsize=(12, 6))
        plt.plot(avg_all['date'], avg_all['wasting'], label='All Wards', marker='o')
        plt.plot(avg_filtered['date'], avg_filtered['wasting'], label='Filtered Wards', marker='x', linestyle='--')
        plt.axhline(y=0, color='black', linestyle='--')
        plt.title('Average Wasting Prevalence Over Time')
        plt.xlabel('Date')
        plt.ylabel('Average Wasting')
        plt.legend()
        plt.tight_layout()
                                    
        if save_plots:
            path = os.path.join(summary_folder, "average_wasting_prevalence.png")
            plt.savefig(path)
            print(f"Saved: {path}")
        if plot:
            plt.show()
        plt.close()
        
        missing = self.wards_shapefile[self.wards_shapefile['Ward'].isin(
            self.muac_data[~self.muac_data['Ward'].isin(self.muac_data_filtered['Ward'])]['Ward']
        )]
        self.wards_final = self.wards_shapefile[
            self.wards_shapefile['Ward'].isin(self.muac_data_filtered['Ward'])
        ]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        missing.plot(ax=ax, color='red', edgecolor='black')
        self.wards_final.plot(ax=ax, color='lightblue', edgecolor='black')
        red_patch = mpatches.Patch(color='red', label='Eliminated Wards')
        blue_patch = mpatches.Patch(color='lightblue', label='Remaining Wards')
        plt.legend(handles=[red_patch, blue_patch])
        plt.title("Wards After Continuity Cleaning")
        plt.tight_layout()
        if save_plots:
            path = os.path.join(summary_folder, "remaining_vs_eliminated_wards.png")
            plt.savefig(path)
            print(f"Saved: {path}")
        if plot:
            plt.show()
        plt.close()


    def reindex_complete_panel(self):
        """
        Insert rows for ward-months with no MUAC collection, so the panel is a
        complete ward x month grid between each ward's first and last
        observation.

        Two reasons. First, pandas shift() is positional, not time-aware: with
        a collection gap, wasting_lag_1 for October holds August's value. In
        the current panel this affected 89 rows at lag 1 and 255 rows (5.2%) at
        lag 3, across 65 of 119 wards, and applied to every lagged covariate.
        Second, the model can then predict for wards that did not report that
        month - in August 2025 five counties submitted nothing, so 28 wards had
        no row and got no forecast at all.

        Inserted rows are flagged is_gap_row so downstream steps can tell "no
        data was collected" apart from "this is a forecast row".
        """
        d = self.muac_data_filtered.copy()
        d["t"] = d["year"] * 12 + d["month"]
        spans = d.groupby(self.polygon_id_col)["t"].agg(["min", "max"])
        grid = pd.concat([
            pd.DataFrame({self.polygon_id_col: wd,
                          "t": range(int(r["min"]), int(r["max"]) + 1)})
            for wd, r in spans.iterrows()], ignore_index=True)
        grid["year"] = (grid["t"] - 1) // 12
        grid["month"] = grid["t"] - grid["year"] * 12

        before = len(d)
        d = grid.merge(d, on=[self.polygon_id_col, "year", "month", "t"], how="left")
        print(f"panel reindexed: {before} -> {len(d)} rows "
              f"({len(d) - before} gap rows inserted)")

        d["is_gap_row"] = d["wasting"].isna()

        # Ward attributes carry forward; static covariates are not merged yet.
        for c in ["County", "SubCounty", "LivelihoodZone"]:
            if c in d.columns:
                d[c] = d.groupby(self.polygon_id_col)[c].ffill().bfill()

        d["date"] = pd.to_datetime(d[["year", "month"]].assign(day=1))
        self.muac_data_filtered = d.drop(columns="t")
        del d, grid, spans
        gc.collect()

    def merge_travel_time(self, generate_plot=True, save_plot=True, plot=False, 
                          output_dir="covariates_graphs", accessiblity_file_name="accessibility_to_cities_2015.csv"):
        """
        Merges travel time data and optionally generates a heatmap.
        """
        print("Merging travel time data...")
        travel_time = pd.read_csv(os.path.join(self.input_path, accessiblity_file_name))
        self.merged_data = pd.merge(self.muac_data_filtered, travel_time, 
                                    on=self.polygon_id_col, how='left')
        del travel_time
        gc.collect()

        if not generate_plot:
            return

        self.wards_shapefile_final = self.wards_shapefile[
            self.wards_shapefile[self.polygon_id_col].isin(self.merged_data[self.polygon_id_col])].copy()
        self.wards_shapefile_final = self.wards_shapefile_final.merge(
            self.merged_data[[self.polygon_id_col, 'travel_time_to_cities_2015']], 
            on=self.polygon_id_col, how='left'
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        self.wards_shapefile_final.plot(column='travel_time_to_cities_2015', cmap='YlOrRd',
                            linewidth=0.8, ax=ax, edgecolor='0.8')

        sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(
            vmin=self.wards_shapefile_final['travel_time_to_cities_2015'].min(),
            vmax=self.wards_shapefile_final['travel_time_to_cities_2015'].max()))
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Travel Time to Nearest City (minutes)')

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()

        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "remoteness_travel_time_2025.png")
            plt.savefig(path)
            print(f"Saved: {path}")
        if plot:
           plt.show()
        plt.close()


    def merge_population_density(self, generate_plot=True, save_plot=True, plot=False, 
                                 output_dir="covariates_graphs",
                                 population_file_name="population_density_2015_2020.csv",
                                 map_year=2020):
        """
        Merges population density data and optionally generates a heatmap.
        """
        print("Merging population density data...")
        pop_den = pd.read_csv(os.path.join(self.input_path, population_file_name))
        self.merged_data = pd.merge(self.merged_data, pop_den, 
                                    on=self.polygon_id_col, how='left')
        del pop_den
        gc.collect()

        if not generate_plot:
            return

        self.wards_shapefile_final = self.wards_shapefile_final.merge(
            self.merged_data[[self.polygon_id_col, f'density_{map_year}']],
              on=self.polygon_id_col, how='left'
        )

        quantiles = np.linspace(0, 1, 10)
        quantile_values = self.wards_shapefile_final[f'density_{map_year}'].quantile(quantiles).to_numpy()
        norm = BoundaryNorm(boundaries=quantile_values, ncolors=256)

        fig, ax = plt.subplots(figsize=(12, 8))
        self.wards_shapefile_final.plot(column=f'density_{map_year}', cmap='YlOrRd',
                            linewidth=0.8, ax=ax, edgecolor='0.8', norm=norm)

        sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=norm)
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax, ticks=quantile_values) 
        cbar.ax.set_yticklabels([f'{q:.2f}' for q in quantile_values])
        cbar.set_label(f'Population Density in {map_year} (Quantile-Based)')

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()

        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"density_{map_year}.png")
            plt.savefig(path)
            print(f"Saved: {path}")
        if plot:
           plt.show()
        plt.close()


    #----------------------------------------------------------
    # Standardised index helpers (used by merge_precipitation_data)
    #----------------------------------------------------------
    def _compute_spi(self, full_data, baseline_years, analysis_years,
                     value_col='precip_total', scale=1):
        """
        SPI per ward and calendar month. scale=3 accumulates the current and two
        preceding months - the usual drought-monitoring horizon and closer to the
        timescale on which rainfall deficits reach nutrition outcomes.
        Gamma parameters are fitted on the baseline window only.
        """
        out_col = f"spi_{scale}"
        df = full_data.copy()
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df = df.sort_values([self.polygon_id_col, 'date'])

        if scale > 1:
            df[value_col] = (df.groupby(self.polygon_id_col)[value_col]
                             .transform(lambda s: s.rolling(scale, min_periods=scale).sum()))

        base = df[(df['year'] >= baseline_years[0]) & (df['year'] <= baseline_years[1])]
        analysis = df[df['year'] >= analysis_years[0]].copy()
        del df
        gc.collect()

        analysis[out_col] = np.nan
        analysis[f'{out_col}_method'] = 'no_baseline'

        groups = {k: v[value_col].to_numpy()
                  for k, v in base.groupby([self.polygon_id_col, 'month'], sort=False)}
        del base
        gc.collect()

        for key, idx in analysis.groupby([self.polygon_id_col, 'month'],
                                         sort=False).groups.items():
            b = groups.get(key)
            if b is None:
                continue
            vals, method = _spi_cell(b, analysis.loc[idx, value_col].to_numpy())
            analysis.loc[idx, out_col] = vals
            analysis.loc[idx, f'{out_col}_method'] = method

        v = analysis[out_col]
        print(f"  {out_col}: n={v.notna().sum()}, NaN={v.isna().mean()*100:.1f}%, "
              f"sd={v.std():.3f}, range [{v.min():.2f}, {v.max():.2f}]")
        print(f"    methods: {analysis[f'{out_col}_method'].value_counts().to_dict()}")
        assert np.isfinite(v.dropna()).all(), f"non-finite values in {out_col}"

        res = analysis[[self.polygon_id_col, 'year', 'month',
                        out_col, f'{out_col}_method']].copy()
        del analysis, groups
        gc.collect()
        return res

    def _normal_score_cols(self, full_data, baseline_years, analysis_years,
                           value_cols, suffix='_ns'):
        """
        Empirical normal score for the bounded count variables. These fail the
        z-score the same way precipitation does - dry_days is 30 or 31 in every
        baseline year for a very arid ward, so sigma is zero - but they are
        discrete and bounded, so no parametric family is defensible.
        """
        base = full_data[(full_data['year'] >= baseline_years[0])
                         & (full_data['year'] <= baseline_years[1])]
        analysis = full_data[full_data['year'] >= analysis_years[0]].copy()

        for col in value_cols:
            out_col = f'{col}{suffix}'
            analysis[out_col] = np.nan
            groups = {k: v[col].to_numpy()
                      for k, v in base.groupby([self.polygon_id_col, 'month'], sort=False)}
            for key, idx in analysis.groupby([self.polygon_id_col, 'month'],
                                             sort=False).groups.items():
                b = groups.get(key)
                if b is None:
                    continue
                b = b[np.isfinite(b)]
                if len(b) < MIN_BASELINE_N:
                    continue
                t = analysis.loc[idx, col].to_numpy()
                vals = _normal_score(b, t)
                vals[~np.isfinite(t)] = np.nan
                analysis.loc[idx, out_col] = vals
            del groups
            gc.collect()

            v = analysis[out_col]
            print(f"  {out_col}: NaN={v.isna().mean()*100:.1f}%, sd={v.std():.3f}, "
                  f"range [{v.min():.2f}, {v.max():.2f}]")
            assert np.isfinite(v.dropna()).all(), f"non-finite values in {out_col}"

        res = analysis[[self.polygon_id_col, 'year', 'month']
                       + [f'{c}{suffix}' for c in value_cols]].copy()
        del base, analysis
        gc.collect()
        return res

    def check_no_infinities(self):
        """
        Fails loudly if any numeric column contains inf. XGBoost tolerates NaN
        but not infinity, and inf propagates into every lag and seasonal
        aggregate built from the affected column.
        """
        num = self.merged_data.select_dtypes(include=[np.number])
        bad = []
        for col in num.columns:
            v = pd.to_numeric(num[col], errors='coerce').to_numpy(dtype='float64')
            if np.isinf(v).any():
                bad.append(col)
        if bad:
            print(f"[!] {len(bad)} columns contain inf:")
            print("   ", bad[:20])
            raise ValueError("Infinite values in the modelling dataset")
        print("No infinite values in the modelling dataset.")

    def merge_temperature_data(self, 
                                main_file,
                                new_file,
                                baseline_years,
                                analysis_years,
                                generate_plot=True, save_plot=True, plot=False,
                                output_dir="covariates_graphs"):
        """
        Merges ERA5 temperature data, computes Z-scores, and optionally generates a plot.
        """
        print("Merging ERA5 temperature data...")

        old_path = os.path.join(self.input_path, main_file)
        old_data = pd.read_pickle(old_path)

        if new_file is not None:
            new_path = os.path.join(self.input_path, new_file)
            if os.path.exists(new_path):
                try:
                    new_data = pd.read_pickle(new_path)
                    full_data = pd.concat([old_data, new_data], ignore_index=True)
                    del old_data, new_data
                    gc.collect()
                    print(f"New ERA5 data found and merged with old data.")

                    if 'year' in full_data.columns and 'month' in full_data.columns:
                        full_data['date'] = pd.to_datetime(
                            full_data['year'].astype(str) + '-' + full_data['month'].astype(str) + '-01')
                        start_date = full_data['date'].min().strftime('%Y_%m')
                        end_date = full_data['date'].max().strftime('%Y_%m')
                    else:
                        raise ValueError("Columns 'year' and 'month' must be present in the data.")

                    new_filename = f"era5_temperature_stats_{start_date}_Onwards.pkl"
                    output_path = os.path.join(self.input_path, new_filename)
                    full_data.to_pickle(output_path)
                    print(f"Saved merged dataset to '{output_path}'.")

                    try:
                        os.remove(new_path)
                        print(f"Deleted new file: {new_path}")
                    except Exception as e:
                        print(f"Could not delete new file: {e}")

                except Exception as e:
                    print(f"Error loading new file: {e}. Proceeding with old data only.")
                    full_data = old_data
            else:
                print("No new ERA5 file found. Proceeding with old data only.")
                full_data = old_data
        else:
            print("No new ERA5 file found. Proceeding with old data only.")
            full_data = old_data

        # Remove duplicates
        dupes = full_data.duplicated(subset=[self.polygon_id_col, 'year', 'month'], keep=False)
        if dupes.any():
            print("Duplicates found in temperature data:")
            print(full_data.loc[dupes, [self.polygon_id_col, 'year', 'month', 'avg_temp_month']])
            full_data = full_data.drop_duplicates(
                subset=[self.polygon_id_col, 'year', 'month'], keep='last')

        full_data['date'] = pd.to_datetime(full_data[['year', 'month']].assign(day=1))
        full_data = full_data.sort_values('date').copy()

        baseline = full_data[(
            full_data['year'] >= baseline_years[0]) & (full_data['year'] <= baseline_years[1])]
        analysis = full_data[(full_data['year'] >= analysis_years[0])].copy()

        # Free full_data now that baseline and analysis are sliced
        # NOTE: full_data is deleted AFTER the normal scores are computed -
        # _normal_score_cols needs it.

        # Per WARD and month, not pooled across all wards. Pooling made the
        # z-score encode structural aridity rather than local deviation.
        # Only avg_temp_month keeps the z-score: it is continuous with real
        # spread, so the transform is sound.
        monthly_stats = baseline.groupby([self.polygon_id_col, 'month']).agg(
            avg_temp_longterm=('avg_temp_month', 'mean'),
            avg_temp_std=('avg_temp_month', 'std'),
            baseline_n=('avg_temp_month', 'count')
        ).reset_index()

        # A near-zero spread produces enormous z-scores
        for _c in [c for c in monthly_stats.columns if c.endswith('_std')]:
            monthly_stats[_c] = monthly_stats[_c].where(monthly_stats[_c] > 1e-6)

        del baseline
        gc.collect()

        analysis = pd.merge(analysis, monthly_stats,
                            on=[self.polygon_id_col, 'month'], how='left')
        del monthly_stats
        gc.collect()

        analysis['avg_temp_z_score'] = (analysis['avg_temp_month'] - analysis['avg_temp_longterm']) / analysis['avg_temp_std']

        #------------------------------------------------------
        # The four count variables move to the empirical normal score.
        # hot_days_z_score reached 680 and consec_hot_days_z_score 338: in a
        # ward-month where the temperature threshold is essentially never
        # crossed the baseline is near-constant, so the standard deviation is
        # tiny and the z-score explodes. The 1e-6 floor does not catch it
        # because a std of 0.05 passes and still gives a z in the hundreds.
        # These are bounded discrete counts, so no parametric family is
        # defensible - the same reasoning applied to the precipitation counts.
        #------------------------------------------------------
        print("Computing temperature normal scores...")
        temp_ns = self._normal_score_cols(
            full_data, baseline_years, analysis_years,
            value_cols=['hot_days', 'cold_days',
                        'consec_hot_days', 'consec_cold_days'])
        analysis = analysis.merge(
            temp_ns, on=[self.polygon_id_col, 'year', 'month'], how='left')
        del temp_ns, full_data
        gc.collect()
        analysis['zscore_baseline'] = f"{baseline_years[0]}–{baseline_years[1]}"

        filtered_temp_data = analysis[[self.polygon_id_col, 'year', 'month',
                                        'avg_temp_month', 'avg_temp_z_score',
                                        'hot_days', 'hot_days_ns',
                                        'cold_days', 'cold_days_ns',
                                        'consec_hot_days', 'consec_hot_days_ns',
                                        'consec_cold_days', 'consec_cold_days_ns',
                                        'baseline_n', 'zscore_baseline']].copy()
        del analysis
        gc.collect()

        expected_polygons = self.merged_data[self.polygon_id_col].unique()
        filtered_temp_data = filtered_temp_data[
            filtered_temp_data[self.polygon_id_col].isin(expected_polygons)
        ]
        self.merged_data = pd.merge(self.merged_data, filtered_temp_data,
                                    on=[self.polygon_id_col, 'year', 'month'], how='left')
        print("Temperature data merged successfully.")

        # === Plotting Section ===
        if not generate_plot:
            del filtered_temp_data
            gc.collect()
            return

        avg_temp = (
            filtered_temp_data.groupby(['year', 'month'])
            .agg(average_temp=('avg_temp_z_score', 'mean'))
            .reset_index()
        )
        avg_temp['date'] = pd.to_datetime(avg_temp[['year', 'month']].assign(day=1))

        del filtered_temp_data
        gc.collect()

        plt.figure(figsize=(12, 6))
        plt.plot(avg_temp['date'], avg_temp['average_temp'], marker='o', color='b', label='Average Temp. Z-score')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Longterm normal (y=0)')
        plt.xlabel('Date')
        plt.ylabel('Monthly Temperature Z-score')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"average_temperature_zscore_{analysis_years[0]}_{analysis_years[1]}.png")
            plt.savefig(path)
            print(f"Saved: {path}")
        if plot:
           plt.show()
        plt.close()


    def merge_precipitation_data(self, 
                                main_file="chirps_precipitation_stats_2004_01_Onwards.pkl",
                                new_file=None,
                                baseline_years=(2004, 2020),
                                analysis_years=(2021, 2026),
                                generate_plot=True, save_plot=True, plot=False,
                                output_dir="covariates_graphs"):
        """
        Merges CHIRPS precipitation data, computes Z-scores, and optionally generates plots.
        """
        print("Merging CHIRPS precipitation data...")

        old_path = os.path.join(self.input_path, main_file)
        old_data = pd.read_pickle(old_path)

        if new_file is not None:
            new_path = os.path.join(self.input_path, new_file)
            if os.path.exists(new_path):
                try:
                    new_data = pd.read_pickle(new_path)
                    full_data = pd.concat([old_data, new_data], ignore_index=True)
                    del old_data, new_data
                    gc.collect()
                    print(f"New CHIRPS data found and merged with old data.")

                    if 'year' in full_data.columns and 'month' in full_data.columns:
                        full_data['date'] = pd.to_datetime(
                            full_data['year'].astype(str) + '-' + full_data['month'].astype(str) + '-01')
                        start_date = full_data['date'].min().strftime('%Y_%m')
                        end_date = full_data['date'].max().strftime('%Y_%m')
                    else:
                        raise ValueError("Columns 'year' and 'month' must be present in the data.")

                    new_filename = f"chirps_precipitation_stats_{start_date}_Onwards.pkl"
                    output_path = os.path.join(self.input_path, new_filename)
                    full_data.to_pickle(output_path)
                    print(f"Saved merged dataset to '{output_path}'.")

                    try:
                        os.remove(new_path)
                        print(f"Deleted new file: {new_path}")
                    except Exception as e:
                        print(f"Could not delete new file: {e}")

                except Exception as e:
                    print(f"Error loading new file: {e}. Proceeding with old data only.")
                    full_data = old_data
            else:
                print("New file path provided but file not found. Proceeding with old data only.")
                full_data = old_data
        else:
            print("No new CHIRPS file specified. Proceeding with old data only.")
            full_data = old_data

        dupes = full_data.duplicated(subset=[self.polygon_id_col, 'year', 'month'], keep=False)
        if dupes.any():
            print("Duplicates found in precipitation data:")
            print(full_data.loc[dupes, [self.polygon_id_col, 'year', 'month', 'precip_total']])
            full_data = full_data.drop_duplicates(subset=[self.polygon_id_col, 'year', 'month'], keep='last')

        full_data['date'] = pd.to_datetime(full_data[['year', 'month']].assign(day=1))
        full_data = full_data.sort_values('date').copy()

        analysis = full_data[(full_data['year'] >= analysis_years[0])].copy()

        #------------------------------------------------------
        # Standardised indices
        #   precip_total  -> SPI-1 and SPI-3 (gamma + explicit mass at zero)
        #   count vars    -> empirical normal score
        # consec_wet_days is dropped: 36% of its baseline ward-month cells are
        # zero in more than 80% of years and 7% have zero variance, so it
        # carries almost no information where it is most needed.
        # NOTE: full_data must stay alive until these are computed.
        #------------------------------------------------------
        print("Computing SPI and normal scores...")
        spi1 = self._compute_spi(full_data, baseline_years, analysis_years,
                                 value_col='precip_total', scale=1)
        spi3 = self._compute_spi(full_data, baseline_years, analysis_years,
                                 value_col='precip_total', scale=3)
        ns = self._normal_score_cols(
            full_data, baseline_years, analysis_years,
            value_cols=['wet_days', 'consec_dry_days'])

        del full_data
        gc.collect()

        analysis = (analysis
                    .merge(spi1, on=[self.polygon_id_col, 'year', 'month'], how='left')
                    .merge(spi3, on=[self.polygon_id_col, 'year', 'month'], how='left')
                    .merge(ns, on=[self.polygon_id_col, 'year', 'month'], how='left'))
        del spi1, spi3, ns
        gc.collect()

        analysis['spi_baseline'] = f"{baseline_years[0]}-{baseline_years[1]}"

        # Raw values kept alongside the standardised ones for the fact cards
        filtered_prec_data = analysis[[self.polygon_id_col, 'year', 'month',
                                    'precip_total', 'spi_1', 'spi_3',
                                    'spi_1_method', 'spi_3_method',
                                    'wet_days', 'wet_days_ns',
                                    'consec_dry_days', 'consec_dry_days_ns',
                                    'spi_baseline']].copy()
        del analysis
        gc.collect()

        expected_polygons = self.merged_data[self.polygon_id_col].unique()
        filtered_prec_data = filtered_prec_data[
            filtered_prec_data[self.polygon_id_col].isin(expected_polygons)
        ]

        self.merged_data = pd.merge(self.merged_data, 
                                    filtered_prec_data, 
                                    on=[self.polygon_id_col, 'year', 'month'], 
                                    how='left')
        print("✅ Precipitation data merged successfully.")

        if not generate_plot:
            del filtered_prec_data
            gc.collect()
            return

        # === PLOT 1: Basic Average Precipitation Z-score ===
        avg_precip = (
            filtered_prec_data.groupby(['year', 'month'])
            .agg(avg_precip=('spi_1', 'mean'))
            .reset_index()
        )
        avg_precip['date'] = pd.to_datetime(avg_precip[['year', 'month']].assign(day=1))

        del filtered_prec_data
        gc.collect()

        plt.figure(figsize=(12, 6))
        plt.plot(avg_precip['date'], avg_precip['avg_precip'], marker='o', color='b', label='Average Precipitation Z-score')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Longterm normal (y=0)')
        plt.xlabel('Date')
        plt.ylabel('Average SPI')
        plt.title('Average SPI Over Time')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, 
                                f"average_precipitation_zscore_{analysis_years[0]}_{analysis_years[1]}.png")
            plt.savefig(path)
            print(f"📈 Saved: {path}")
        plt.close()

        # === PLOT 2: Enhanced Precip Z-score vs Wasting ===
        required_cols = {"year", "month", "wasting_count", "wasting_risk_count"}
        if not required_cols.issubset(self.merged_data.columns):
            missing = required_cols - set(self.merged_data.columns)
            print(f"[!] Skipping enhanced precipitation plot - missing columns: {missing}")
            return

        wasting_by_month = (
            self.merged_data.groupby(["year", "month"], as_index=False)
            .agg(
                wasting_count=("wasting_count", "sum"),
                wasting_risk_count=("wasting_risk_count", "sum"),
            )
        )
        wasting_by_month["date"] = pd.to_datetime(wasting_by_month[["year", "month"]].assign(day=1))

        pm = pd.merge(
            avg_precip[["date", "year", "month", "avg_precip"]],
            wasting_by_month[["date", "year", "month", "wasting_count", "wasting_risk_count"]],
            on=["year", "month", "date"],
            how="inner",
        ).sort_values("date")

        C_PRECIP = "#1f77b4"
        C_ZERO   = "#6e6e6e"
        C_ANOM   = "#8c2d04"
        C_LONG   = "#cfe8ff"
        C_SHORT  = "#b2f1e5"
        C_WASTE  = "#e6550d"

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(pm["date"], pm["avg_precip"],
                color=C_PRECIP, linewidth=2, marker="o", markersize=4,
                label="Precipitation Z-score")
        ax1.axhline(y=0, linestyle="--", linewidth=1.2, color=C_ZERO, label="Long-term normal (y=0)")

        neg_mask = pm["avg_precip"] < 0
        if neg_mask.any():
            ax1.fill_between(mdates.date2num(pm["date"]),
                            pm["avg_precip"].to_numpy(), 0,
                            where=neg_mask.to_numpy(), alpha=0.25, color=C_ANOM,
                            label="Precipitation anomaly: precip below longterm average for that month")
        else:
            anomaly_patch = mpatches.Patch(alpha=0.25, color=C_ANOM,
                                        label="Precipitation anomaly: precip below longterm average for that month")

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Precipitation Z-score")

        xmin, xmax = pm["date"].min(), pm["date"].max()
        years = np.sort(pm["date"].dt.year.unique())
        added_long, added_short = False, False
        for y in years:
            a0, b0 = max(pd.Timestamp(y,3,1), xmin), min(pd.Timestamp(y,6,1), xmax)
            if a0 < b0:
                ax1.axvspan(a0, b0, color=C_LONG, alpha=0.35,
                            label="Long rains (Mar–May)" if not added_long else None)
                added_long = True
            a1, b1 = max(pd.Timestamp(y,10,1), xmin), min(pd.Timestamp(y+1,1,1), xmax)
            if a1 < b1:
                ax1.axvspan(a1, b1, color=C_SHORT, alpha=0.35,
                            label="Short rains (Oct–Dec)" if not added_short else None)
                added_short = True

        ax2 = ax1.twinx()
        ax2.set_ylabel("Counts")
        ax2.plot(pm["date"], pm["wasting_count"],
                color=C_WASTE, linewidth=2.2, marker="s", markersize=4,
                linestyle="-", label="Wasting count")

        fig.autofmt_xdate()

        lines1, labels1 = ax1.get_legend_handles_labels()
        if not neg_mask.any():
            lines1.append(anomaly_patch)
            labels1.append(anomaly_patch.get_label())
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.title("Precipitation Z-score vs Wasting Over Time")
        plt.tight_layout()

        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(
                output_dir,
                f"precip_zscore_vs_wasting_{analysis_years[0]}_{analysis_years[1]}.png",
            )
            plt.savefig(out_path, bbox_inches="tight")
            print(f"📈 Saved: {out_path}")
        plt.close()


    def merge_ndvi_evi_data(self, 
                            main_file="modis_ndvi_evi_stats_2016_01_Onwards.pkl",
                            new_file=None,
                            baseline_years=(2016, 2020),
                            analysis_years=(2021, 2026),
                            generate_plot=True, save_plot=True, plot=False,
                            output_dir="covariates_graphs"):
        """
        Merges MODIS NDVI/EVI data, computes Z-scores, and optionally generates plots.
        """
        print("Merging MODIS NDVI/EVI data...")

        old_path = os.path.join(self.input_path, main_file)
        old_data = pd.read_pickle(old_path)

        if new_file is not None:
            new_path = os.path.join(self.input_path, new_file)
            if os.path.exists(new_path):
                try:
                    new_data = pd.read_pickle(new_path)
                    full_data = pd.concat([old_data, new_data], ignore_index=True)
                    del old_data, new_data
                    gc.collect()
                    print("New NDVI/EVI data found and merged with old data.")

                    if 'year' in full_data.columns and 'month' in full_data.columns:
                        full_data['date'] = pd.to_datetime(
                            full_data['year'].astype(str) + '-' + full_data['month'].astype(str) + '-01')
                        start_date = full_data['date'].min().strftime('%Y_%m')
                        end_date = full_data['date'].max().strftime('%Y_%m')
                    else:
                        raise ValueError("Columns 'year' and 'month' must be present in the data.")

                    new_filename = f"modis_ndvi_evi_stats_{start_date}_Onwards.pkl"
                    output_path = os.path.join(self.input_path, new_filename)
                    full_data.to_pickle(output_path)
                    print(f"Saved merged dataset to '{output_path}'.")

                    try:
                        os.remove(new_path)
                        print(f"Deleted new file: {new_path}")
                    except Exception as e:
                        print(f"Could not delete new file: {e}")

                except Exception as e:
                    print(f"Error loading new file: {e}. Proceeding with old data only.")
                    full_data = old_data
            else:
                print("New file path provided but file not found. Proceeding with old data only.")
                full_data = old_data
        else:
            print("No new NDVI/EVI file specified. Proceeding with old data only.")
            full_data = old_data

        dupes = full_data.duplicated(subset=[self.polygon_id_col, 'year', 'month'], keep=False)
        if dupes.any():
            print("Duplicates found in NDVI/EVI data:")
            print(full_data.loc[dupes, [self.polygon_id_col, 'year', 'month', 'EVI_mean']])
            full_data = full_data.drop_duplicates(subset=[self.polygon_id_col, 'year', 'month'], keep='last')

        full_data['date'] = pd.to_datetime(full_data[['year', 'month']].assign(day=1))
        full_data = full_data.sort_values('date').copy()

        baseline = full_data[(full_data['year'] >= baseline_years[0]) & (full_data['year'] <= baseline_years[1])]
        analysis = full_data[(full_data['year'] >= analysis_years[0])].copy()

        del full_data
        gc.collect()

        # Per WARD and month. Only 5 baseline years here, so veg_baseline_n is
        # carried through for the fact-card layer.
        monthly_stats = baseline.groupby([self.polygon_id_col, 'month']).agg(
            NDVI_longterm=('NDVI_mean', 'mean'),
            NDVI_std=('NDVI_mean', 'std'),
            EVI_longterm=('EVI_mean', 'mean'),
            EVI_std=('EVI_mean', 'std'),
            veg_baseline_n=('EVI_mean', 'count')
        ).reset_index()

        for _c in ['NDVI_std', 'EVI_std']:
            monthly_stats[_c] = monthly_stats[_c].where(monthly_stats[_c] > 1e-6)

        del baseline
        gc.collect()

        analysis = pd.merge(analysis, monthly_stats,
                            on=[self.polygon_id_col, 'month'], how='left')
        del monthly_stats
        gc.collect()

        analysis['NDVI_z_score'] = (analysis['NDVI_mean'] - analysis['NDVI_longterm']) / analysis['NDVI_std']
        analysis['EVI_z_score'] = (analysis['EVI_mean'] - analysis['EVI_longterm']) / analysis['EVI_std']
        analysis['zscore_baseline_EVI'] = f"{baseline_years[0]}–{baseline_years[1]}"

        filtered_evi_data = analysis[[self.polygon_id_col, 'year', 'month',
                                'NDVI_mean', 'NDVI_z_score',
                                'EVI_mean', 'EVI_z_score',
                                'veg_baseline_n', 'zscore_baseline_EVI']].copy()
        del analysis
        gc.collect()

        expected_polygons = self.merged_data[self.polygon_id_col].unique()
        filtered_evi_data = filtered_evi_data[
            filtered_evi_data[self.polygon_id_col].isin(expected_polygons)
        ]
        self.merged_data = pd.merge(self.merged_data, 
                                    filtered_evi_data, 
                                    on=[self.polygon_id_col, 'year', 'month'], 
                                    how='left')
        print("✅ NDVI/EVI data merged successfully.")

        if not generate_plot:
            del filtered_evi_data
            gc.collect()
            return

        avg_z = (
            filtered_evi_data.groupby(['year', 'month'])
            .agg(avg_NDVI_z_score=('NDVI_z_score', 'mean'),
                avg_EVI_z_score=('EVI_z_score', 'mean'))
            .reset_index()
        )
        avg_z['date'] = pd.to_datetime(avg_z[['year', 'month']].assign(day=1))

        del filtered_evi_data
        gc.collect()

        plt.figure(figsize=(12, 6))
        plt.plot(avg_z['date'], avg_z['avg_NDVI_z_score'], marker='o', color='green', label='Average NDVI Z-Score')
        plt.plot(avg_z['date'], avg_z['avg_EVI_z_score'], marker='x', color='blue', label='Average EVI Z-Score')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Midterm normal (y=0)')
        plt.xlabel('Date')
        plt.ylabel('Average Z-Score')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"average_ndvi_evi_zscores_{analysis_years[0]}_{analysis_years[1]}.png")
            plt.savefig(path)
            print(f"📈 Saved: {path}")
        plt.close()

        if "wasting_count" not in self.merged_data.columns:
            print("[!] Skipping enhanced EVI plot - missing wasting_count column")
            return

        wasting_by_month = (
            self.merged_data.groupby(["year","month"], as_index=False)
            .agg(wasting_count=("wasting_count","sum"))
        )
        wasting_by_month["date"] = pd.to_datetime(wasting_by_month[["year","month"]].assign(day=1))

        pm_evi = (
            avg_z.merge(
                wasting_by_month[["year","month","date","wasting_count"]],
                on=["year","month","date"], how="inner"
            )
            .sort_values("date")
        )

        C_EVI   = "#1f77b4"
        C_ZERO  = "#6e6e6e"
        C_ANOM  = "#8c2d04"
        C_LONG  = "#cfe8ff"
        C_SHORT = "#b2f1e5"
        C_WASTE = "#e6550d"

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(pm_evi["date"], pm_evi["avg_EVI_z_score"],
                color=C_EVI, linewidth=2, marker="o", markersize=4,
                label="Average EVI Z-Score")
        ax1.axhline(y=0, linestyle="--", linewidth=1.2, color=C_ZERO, label="Long-term normal (y=0)")

        neg_mask = (pm_evi["avg_EVI_z_score"] < 0).to_numpy()
        if neg_mask.any():
            ax1.fill_between(
                mdates.date2num(pm_evi["date"]),
                pm_evi["avg_EVI_z_score"].to_numpy(), 0,
                where=neg_mask, alpha=0.25, color=C_ANOM,
                label="EVI anomaly: EVI below longterm average for that month",
            )
        else:
            anomaly_patch = mpatches.Patch(alpha=0.25, color=C_ANOM,
                                        label="EVI anomaly: EVI below longterm average for that month")

        xmin, xmax = pm_evi["date"].min(), pm_evi["date"].max()
        years = np.sort(pm_evi["date"].dt.year.unique())
        added_long = added_short = False
        for y in years:
            a0, b0 = max(pd.Timestamp(y,3,1), xmin), min(pd.Timestamp(y,6,1), xmax)
            if a0 < b0:
                ax1.axvspan(a0, b0, color=C_LONG, alpha=0.35,
                            label="Long rains (Mar–May)" if not added_long else None)
                added_long = True
            a1, b1 = max(pd.Timestamp(y,10,1), xmin), min(pd.Timestamp(y+1,1,1), xmax)
            if a1 < b1:
                ax1.axvspan(a1, b1, color=C_SHORT, alpha=0.35,
                            label="Short rains (Oct–Dec)" if not added_short else None)
                added_short = True

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Average EVI Z-Score")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Wasting count")
        ax2.plot(pm_evi["date"], pm_evi["wasting_count"],
                color=C_WASTE, linewidth=2.2, marker="D", markersize=4,
                label="Wasting count")

        lines1, labels1 = ax1.get_legend_handles_labels()
        if not neg_mask.any():
            lines1.append(anomaly_patch)
            labels1.append(anomaly_patch.get_label())
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        plt.title("EVI/NDVI Z-scores vs Wasting Over Time")
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(
                output_dir,
                f"evi_ndvi_zscore_vs_wasting_{analysis_years[0]}_{analysis_years[1]}.png"
            )
            plt.savefig(path, bbox_inches="tight")
            print(f"📈 Saved: {path}")
        plt.close()


    def merge_ndvi_evi_by_land_use(self,
                                    main_file="modis_ndvi_with_land_use_evi_stats_2018_01_to_2024_08.pkl",
                                    new_file=None,
                                    baseline_years=(2016, 2020),
                                    analysis_years=(2021, 2026),
                                    generate_plot=True, save_plot=True, plot=False,
                                    output_dir="covariates_graphs"):
        """
        Merges NDVI/EVI data by land use class, computes Z-scores.
        """
        print("Merging NDVI/EVI data by land use class...")

        old_path = os.path.join(self.input_path, main_file)
        old_data = pd.read_pickle(old_path)

        if new_file is not None:
            new_path = os.path.join(self.input_path, new_file)
            if os.path.exists(new_path):
                try:
                    new_data = pd.read_pickle(new_path)
                    full_data = pd.concat([old_data, new_data], ignore_index=True)
                    del old_data, new_data
                    gc.collect()
                    print("New land use NDVI/EVI data found and merged with old data.")

                    if 'year' in full_data.columns and 'month' in full_data.columns:
                        full_data['date'] = pd.to_datetime(
                            full_data['year'].astype(str) + '-' + full_data['month'].astype(str) + '-01')
                        start_date = full_data['date'].min().strftime('%Y_%m')
                        end_date = full_data['date'].max().strftime('%Y_%m')
                    else:
                        raise ValueError("Columns 'year' and 'month' must be present in the data.")

                    new_filename = f"modis_evi_by_land_use_stats_{start_date}_Onwards.pkl"
                    output_path = os.path.join(self.input_path, new_filename)
                    full_data.to_pickle(output_path)
                    print(f"Saved merged dataset to '{output_path}'.")

                    try:
                        os.remove(new_path)
                        print(f"Deleted new file: {new_path}")
                    except Exception as e:
                        print(f"Could not delete new file: {e}")

                except Exception as e:
                    print(f"Error loading new file: {e}. Proceeding with old data only.")
                    full_data = old_data
            else:
                print("New file path provided but file not found. Proceeding with old data only.")
                full_data = old_data
        else:
            print("No new file found. Proceeding with old data only.")
            full_data = old_data

        dupes = full_data.duplicated(subset=[self.polygon_id_col, 'year', 'month', 'land_use_class'], keep=False)
        if dupes.any():
            print("Duplicates found in land use NDVI/EVI data:")
            print(full_data.loc[dupes, [self.polygon_id_col, 'year', 'month', 'land_use_class', 'EVI']])
            full_data = full_data.drop_duplicates(subset=[self.polygon_id_col, 'year', 'month', 'land_use_class'], keep='last')

        full_data['date'] = pd.to_datetime(full_data[['year', 'month']].assign(day=1))

        baseline = full_data[(full_data['year'] >= baseline_years[0]) & (full_data['year'] <= baseline_years[1])]
        analysis = full_data[full_data['year'] >= analysis_years[0]].copy()

        del full_data
        gc.collect()

        longterm_stats = baseline.groupby(
            [self.polygon_id_col, 'land_use_class', 'month']).agg(
            NDVI_longterm_mean=('NDVI', 'mean'),
            NDVI_longterm_std=('NDVI', 'std'),
            NDVI_baseline_n=('NDVI', 'count'),
            EVI_longterm_mean=('EVI', 'mean'),
            EVI_longterm_std=('EVI', 'std'),
            EVI_baseline_n=('EVI', 'count'),
        ).reset_index()

        # A ward-class-month baseline with 2 observations gives a std with one
        # degree of freedom: if the two values happen to be close the std is
        # near zero and the z-score explodes (Kauwi grass reached 19,000).
        # An absolute floor does not catch this because EVI is stored x10000,
        # so a std of 0.06 against a mean of 2831 passes any small threshold.
        # Cloud masking leaves many cells with few valid months.
        MIN_LU_BASELINE_N = 3      # of a possible 5 (2016-2020)
        MIN_CV = 0.01              # std must be at least 1% of the mean

        for _var in ['NDVI', 'EVI']:
            _n = longterm_stats[f'{_var}_baseline_n']
            _cv = (longterm_stats[f'{_var}_longterm_std']
                   / longterm_stats[f'{_var}_longterm_mean'].abs().replace(0, np.nan))
            _drop = (_n < MIN_LU_BASELINE_N) | (_cv < MIN_CV)
            longterm_stats.loc[_drop, f'{_var}_longterm_std'] = np.nan
            print(f"  {_var} by land use: {int(_drop.sum())} of {len(longterm_stats)} "
                  f"baseline cells gated ({_drop.mean()*100:.1f}%)")

        del baseline
        gc.collect()

        analysis = pd.merge(analysis, longterm_stats,
                            on=[self.polygon_id_col, 'land_use_class', 'month'],
                            how='left')
        del longterm_stats
        gc.collect()

        analysis['NDVI_z_score'] = (analysis['NDVI'] - analysis['NDVI_longterm_mean']) / analysis['NDVI_longterm_std']
        analysis['EVI_z_score'] = (analysis['EVI'] - analysis['EVI_longterm_mean']) / analysis['EVI_longterm_std']
        analysis['zscore_baseline'] = f"{baseline_years[0]}–{baseline_years[1]}"

        pivot = analysis.pivot_table(
            index=[self.polygon_id_col, 'year', 'month'],
            columns='land_use_class',
            values=['NDVI', 'EVI', 'NDVI_z_score', 'EVI_z_score']
        )
        pivot.columns = [f"{var}_{cls}" for var, cls in pivot.columns]
        pivot.reset_index(inplace=True)

        del analysis
        gc.collect()

        expected_polygons = self.merged_data[self.polygon_id_col].unique()
        filtered_pivot_data = pivot[pivot[self.polygon_id_col].isin(expected_polygons)].copy()

        del pivot
        gc.collect()

        self.merged_data = pd.merge(self.merged_data, filtered_pivot_data, 
                                    on=[self.polygon_id_col, 'year', 'month'], 
                                    how='left')
        print("✅ NDVI/EVI by land use class merged.")

        if not generate_plot:
            del filtered_pivot_data
            gc.collect()
            return

        avg_z = filtered_pivot_data.groupby(['year', 'month']).agg(
            avg_EVI_z_score_grass=('EVI_z_score_grass', 'mean'),
            avg_EVI_z_score_shrub_and_scrub=('EVI_z_score_shrub_and_scrub', 'mean'),
            avg_EVI_z_score_trees=('EVI_z_score_trees', 'mean'),
            avg_EVI_z_score_crops=('EVI_z_score_crops', 'mean')
        ).reset_index()
        avg_z['date'] = pd.to_datetime(avg_z[['year', 'month']].assign(day=1))

        del filtered_pivot_data
        gc.collect()

        plt.figure(figsize=(12, 6))
        plt.plot(avg_z['date'], avg_z['avg_EVI_z_score_grass'], label='Grass', marker='o', color='green')
        plt.plot(avg_z['date'], avg_z['avg_EVI_z_score_shrub_and_scrub'], label='Shrub & Scrub', marker='x', color='brown')
        plt.plot(avg_z['date'], avg_z['avg_EVI_z_score_trees'], label='Trees', marker='s', color='blue')
        plt.plot(avg_z['date'], avg_z['avg_EVI_z_score_crops'], label='Crops', marker='d', color='orange')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Midterm normal (y=0)')
        plt.xlabel('Date')
        plt.ylabel('Average EVI Z-Score')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"evi_zscore_by_land_use_{analysis_years[0]}_{analysis_years[1]}.png")
            plt.savefig(path)
            print(f"📈 Saved: {path}")
        if plot:
           plt.show()
        plt.close()


    def merge_land_use_data(self, 
                            main_file="land_use_pct_stats_2007_01_to_2010_12.pkl",
                            new_file=None,
                            generate_plot=True, save_plot=True, plot=True,
                            output_dir="covariates_graphs"):
        """
        Merges land use percentage data, pivots to wide format.
        """
        print("Merging land use data...")

        old_path = os.path.join(self.input_path, main_file)
        old_data = pd.read_pickle(old_path)

        if new_file is not None:
            new_path = os.path.join(self.input_path, new_file)
            if os.path.exists(new_path):
                try:
                    new_data = pd.read_pickle(new_path)
                    full_data = pd.concat([old_data, new_data], ignore_index=True)
                    del old_data, new_data
                    gc.collect()
                    print("New land use data found and merged with old data.")

                    if 'year' in full_data.columns and 'month' in full_data.columns:
                        full_data['date'] = pd.to_datetime(
                            full_data['year'].astype(str) + '-' + full_data['month'].astype(str) + '-01')
                        start_date = full_data['date'].min().strftime('%Y_%m')
                        end_date = full_data['date'].max().strftime('%Y_%m')
                    else:
                        raise ValueError("Columns 'year' and 'month' must be present in the data.")

                    new_filename = f"land_use_pct_stats_{start_date}_Onwards.pkl"
                    output_path = os.path.join(self.input_path, new_filename)
                    full_data.to_pickle(output_path)
                    print(f"Saved merged dataset to '{output_path}'.")

                    try:
                        os.remove(new_path)
                        print(f"Deleted new file: {new_path}")
                    except Exception as e:
                        print(f"Could not delete new file: {e}")

                except Exception as e:
                    print(f"Error loading new file: {e}. Proceeding with old data only.")
                    full_data = old_data
            else:
                print("New file path provided but file not found. Proceeding with old data only.")
                full_data = old_data
        else:
            print("No new land use file found. Proceeding with old data only.")
            full_data = old_data

        dupes = full_data.duplicated(
            subset=[self.polygon_id_col, 'year', 'month', 'land_use_name'], keep=False)
        if dupes.any():
            print("Duplicates found in land use data:")
            print(full_data.loc[dupes, [self.polygon_id_col, 'year', 
                                        'month', 'land_use_name', 'land_use_percentage']])
            full_data = full_data.drop_duplicates(
                subset=[self.polygon_id_col, 'year', 'month', 'land_use_name'], keep='last')

        full_data['month'] = pd.to_numeric(full_data['month'], errors='coerce').fillna(1).astype(int)
        full_data['date'] = pd.to_datetime(full_data[['year', 'month']].assign(day=1), errors='coerce')

        wide_data = full_data.pivot_table(
            index=[self.polygon_id_col, 'year', 'month', 'date'],
            columns='land_use_name',
            values='land_use_percentage',
            fill_value=0
        ).reset_index()
        wide_data.columns.name = None

        del full_data
        gc.collect()

        merge_data = wide_data.drop(columns=['date'])

        expected_polygons = self.merged_data[self.polygon_id_col].unique()
        merge_data = merge_data[merge_data[self.polygon_id_col].isin(expected_polygons)]

        self.merged_data = pd.merge(
            self.merged_data,
            merge_data,
            on=[self.polygon_id_col, 'year', 'month'],
            how='left'
        )

        del merge_data
        gc.collect()

        print("Land use data merged successfully.")

        self.merged_data['date'] = pd.to_datetime(
            self.merged_data['year'].astype(int).astype(str) + '-' +
            self.merged_data['month'].astype(int).astype(str).str.zfill(2) + '-01',
            errors='coerce'
        )

        if not generate_plot:
            del wide_data
            gc.collect()
            return

        land_use_classes = [col for col in wide_data.columns if col not in [
            'year', 'month', 'date', self.polygon_id_col]]

        if land_use_classes:
            avg_land_use = wide_data.groupby('date')[land_use_classes].mean().reset_index()
            avg_land_use.columns = ['date'] + [f"avg_{c}" for c in land_use_classes]

            del wide_data
            gc.collect()

            plt.figure(figsize=(12, 6))
            for col in avg_land_use.columns:
                if col != 'date':
                    plt.plot(avg_land_use['date'], avg_land_use[col], label=col.replace("avg_", "").capitalize())

            plt.xlabel('Date')
            plt.ylabel('Average % Area')
            plt.title('Average Land Use Percentages Over Time')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()

            if save_plot:
                os.makedirs(output_dir, exist_ok=True)
                path = os.path.join(output_dir, "land_use_avg_pct_over_time.png")
                plt.savefig(path)
                print(f"Saved: {path}")
            if plot:
               plt.show()
            plt.close()


    def merge_conflict_data(self,
                            conflict_100km_old="ACLED_conflict_12m_running_sum_by_polygon_100kmcutoff_2021_08_to_2025_02.pkl",
                            conflict_100km_new=None,
                            conflict_500km_old="ACLED_conflict_12m_running_sum_by_polygon_500kmcutoff_2005_2010.csv",
                            conflict_500km_new=None,
                            filter_from_year=2021,
                            generate_plot=True, save_plot=True, plot=True,
                            output_dir="covariates_graphs"):
        """
        Merges conflict data (100km and 500km).
        """
        print("Merging conflict data (100km and 500km radius)...")

        # --- 100km Conflict Data ---
        c100_old = pd.read_pickle(os.path.join(self.input_path, conflict_100km_old))

        if conflict_100km_new is not None:
            c100_path_new = os.path.join(self.input_path, conflict_100km_new)
            if os.path.exists(c100_path_new):
                try:
                    c100_new = pd.read_pickle(c100_path_new)
                    c100 = pd.concat([c100_old, c100_new], ignore_index=True)
                    del c100_old, c100_new
                    gc.collect()
                    print("New 100km conflict data found and merged.")

                    if 'year' in c100.columns and 'month' in c100.columns:
                        c100['date'] = pd.to_datetime(
                            c100['year'].astype(str) + '-' + c100['month'].astype(str) + '-01')
                        start_date = c100['date'].min().strftime('%Y_%m')
                        end_date = c100['date'].max().strftime('%Y_%m')
                    else:
                        raise ValueError("Columns 'year' and 'month' must be present in the data.")

                    new_filename = f"ACLED_conflict_12m_100kmcutoff_{start_date}_Onwards.pkl"
                    output_path = os.path.join(self.input_path, new_filename)
                    c100.to_pickle(output_path)
                    print(f"Saved merged dataset to '{output_path}'.")

                    try:
                        os.remove(c100_path_new)
                        print(f"Deleted new file: {c100_path_new}")
                    except Exception as e:
                        print(f"Could not delete new file: {e}")

                except Exception as e:
                    print(f"Error loading new 100km file: {e}. Using only old data.")
                    c100 = c100_old
            else:
                print("No new 100km conflict file found. Using only old data.")
                c100 = c100_old
        else:
            print("No new 100km conflict file found. Using only old data.")
            c100 = c100_old

        dupes = c100.duplicated(subset=[self.polygon_id_col, 'year', 'month'], keep=False)
        if dupes.any():
            print("Duplicates found in 100km conflict data:")
            print(c100.loc[dupes, [self.polygon_id_col, 'year', 'month', 'conflict_previous_12m']])
            c100 = c100.drop_duplicates(subset=[self.polygon_id_col, 'year', 'month'], keep='last')

        c100 = c100.rename(columns={
            'monthly_conflicts_dis_w': 'conflict_100km',
            'monthly_fatalities_dis_w': 'fatalities_100km',
            'conflict_previous_12m': 'conflict_12m_100km',
            'fatalities_previous_12m': 'fatalities_12m_100km'
        })
        c100_filtered = c100[c100['year'] >= filter_from_year].copy()
        del c100
        gc.collect()

        if 'date' in c100_filtered.columns:
            c100_filtered = c100_filtered.drop(columns=['date'])

        expected_polygons = self.merged_data[self.polygon_id_col].unique()
        c100_filtered = c100_filtered[c100_filtered[self.polygon_id_col].isin(expected_polygons)]

        self.merged_data = pd.merge(
            self.merged_data, c100_filtered,
            on=[self.polygon_id_col, 'year', 'month'],
            how='left'
        )
        del c100_filtered
        gc.collect()

        # --- 500km Conflict Data ---
        c500_old = pd.read_pickle(os.path.join(self.input_path, conflict_500km_old))

        if conflict_500km_new is not None:
            c500_path_new = os.path.join(self.input_path, conflict_500km_new)
            if os.path.exists(c500_path_new):
                try:
                    c500_new = pd.read_pickle(c500_path_new)
                    c500 = pd.concat([c500_old, c500_new], ignore_index=True)
                    del c500_old, c500_new
                    gc.collect()
                    print("New 500km conflict data found and merged.")

                    if 'year' in c500.columns and 'month' in c500.columns:
                        c500['date'] = pd.to_datetime(
                            c500['year'].astype(str) + '-' + c500['month'].astype(str) + '-01')
                        start_date = c500['date'].min().strftime('%Y_%m')
                        end_date = c500['date'].max().strftime('%Y_%m')
                    else:
                        raise ValueError("Columns 'year' and 'month' must be present in the data.")

                    new_filename = f"ACLED_conflict_12m_500kmcutoff_{start_date}_Onwards.pkl"
                    output_path = os.path.join(self.input_path, new_filename)
                    c500.to_pickle(output_path)
                    print(f"Saved merged dataset to '{output_path}'.")

                    try:
                        os.remove(c500_path_new)
                        print(f"Deleted new file: {c500_path_new}")
                    except Exception as e:
                        print(f"Could not delete new file: {e}")

                except Exception as e:
                    print(f"Error loading new 500km file: {e}. Using only old data.")
                    c500 = c500_old
            else:
                print("No new 500km conflict file found. Using only old data.")
                c500 = c500_old
        else:
            print("No new 500km conflict file found. Using only old data.")
            c500 = c500_old

        dupes = c500.duplicated(subset=[self.polygon_id_col, 'year', 'month'], keep=False)
        if dupes.any():
            print("Duplicates found in 500km conflict data:")
            print(c500.loc[dupes, [self.polygon_id_col, 'year', 'month', 'conflict_previous_12m']])
            c500 = c500.drop_duplicates(subset=[self.polygon_id_col, 'year', 'month'], keep='last')

        c500 = c500.rename(columns={
            'monthly_conflicts_dis_w': 'conflict_500km',
            'monthly_fatalities_dis_w': 'fatalities_500km',
            'conflict_previous_12m': 'conflict_12m_500km',
            'fatalities_previous_12m': 'fatalities_12m_500km'
        })
        c500_filtered = c500[c500['year'] >= filter_from_year].copy()
        del c500
        gc.collect()

        if 'date' in c500_filtered.columns:
            c500_filtered = c500_filtered.drop(columns=['date'])

        expected_polygons = self.merged_data[self.polygon_id_col].unique()
        c500_filtered = c500_filtered[c500_filtered[self.polygon_id_col].isin(expected_polygons)]

        self.merged_data = pd.merge(
            self.merged_data, c500_filtered,
            on=[self.polygon_id_col, 'year', 'month'],
            how='left'
        )
        del c500_filtered
        gc.collect()

        print("Conflict data merged successfully.")

        self.merged_data['date'] = pd.to_datetime(
            self.merged_data['year'].astype(int).astype(str) + '-' +
            self.merged_data['month'].astype(int).astype(str).str.zfill(2) + '-01',
            errors='coerce'
        )

        if not generate_plot:
            return

        avg_fatalities = (
            self.merged_data.groupby(['year', 'month'])
            .agg(fatalities=('fatalities_500km', 'mean'))
            .reset_index()
        )
        avg_conflict = (
            self.merged_data.groupby(['year', 'month'])
            .agg(conflict=('conflict_500km', 'mean'))
            .reset_index()
        )

        conflict_plot_data = pd.merge(avg_conflict, avg_fatalities, on=['year', 'month'])
        conflict_plot_data['date'] = pd.to_datetime(conflict_plot_data[['year', 'month']].assign(day=1))

        plt.figure(figsize=(12, 6))
        plt.plot(conflict_plot_data['date'], conflict_plot_data['conflict'], marker='o', color='b', 
                label='Avg. 12-month Cumulative Conflict (500km)')
        plt.plot(conflict_plot_data['date'], conflict_plot_data['fatalities'], marker='x', linestyle='--', color='r', 
                label='Avg. 12-month Cumulative Fatalities (500km)')
        plt.xlabel('Date')
        plt.ylabel('Average Cumulative Value')
        plt.title('Conflict and Fatalities Over Time (500km Radius)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "conflict_fatalities_trends.png")
            plt.savefig(path)
            print(f"📈 Saved: {path}")
        if plot:
           plt.show()
        plt.close()


    def extend_future_muac_periods(self, months_forward=3):
        """
        Extends the time series for each polygon by `months_forward` months.
        """
        print(f"Extending MUAC data {months_forward} months forward for each polygon...")

        if 'date' not in self.merged_data.columns:
            self.merged_data['date'] = pd.to_datetime(
                self.merged_data[['year', 'month']].assign(day=1)
            )

        last_dates = (
            self.merged_data.dropna(subset=['wasting'])
            .groupby(self.polygon_id_col)['date']
            .max()
            .reset_index()
            .rename(columns={'date': 'last_muac_date'})
        )

        full_date_grid = []

        for _, row in last_dates.iterrows():
            pid = row[self.polygon_id_col]
            last_muac_date = row['last_muac_date']

            future_dates = pd.date_range(
                last_muac_date + pd.offsets.MonthBegin(1),
                last_muac_date + pd.offsets.MonthBegin(months_forward),
                freq='MS'
            )

            for d in future_dates:
                full_date_grid.append({
                    self.polygon_id_col: pid,
                    'date': d,
                    'year': d.year,
                    'month': d.month
                })

        future_df = pd.DataFrame(full_date_grid)

        existing = self.merged_data[['Ward', 'date']]
        future_df = future_df.merge(existing, on=[self.polygon_id_col, 'date'], how='left', indicator=True)
        future_df = future_df[future_df['_merge'] == 'left_only'].drop(columns=['_merge'])

        metadata_cols = ['County', 'SubCounty', 'LivelihoodZone']

        ward_metadata = (
            self.merged_data
            .dropna(subset=metadata_cols)
            .drop_duplicates(subset=[self.polygon_id_col])[[self.polygon_id_col] + metadata_cols]
        )

        future_df = future_df.merge(ward_metadata, on=self.polygon_id_col, how='left')

        self.merged_data = pd.concat([self.merged_data, future_df], ignore_index=True)
        print(f"Added {len(future_df)} future rows.")


    def generate_lags_for_dynamic_variables(self, max_lag=12):
        """
        Generates up to `max_lag` months of lags for dynamic variables.
        """
        print(f"Generating lags up to {max_lag} months for dynamic variables...")

        self.merged_data = self.merged_data.sort_values(
            by=[self.polygon_id_col, 'year', 'month']).reset_index(drop=True)

        self.merged_data = self.merged_data.rename(columns={
            'avg_temp_month': 'avg_temp', 
        })

        # Temperature keeps the z-score. Precipitation now uses SPI and the
        # nonparametric standardised index, so its columns are named explicitly
        # rather than built as f"{var}_z_score".
        # Only avg_temp keeps the z-score; the temperature counts use the
        # nonparametric standardised index, like the precipitation counts.
        climate_vars = ['avg_temp']
        temp_ns_cols = ['hot_days_ns', 'cold_days_ns',
                        'consec_hot_days_ns', 'consec_cold_days_ns']
        precip_cols = ['spi_1', 'spi_3', 'wet_days_ns', 'consec_dry_days_ns']
        conflict_vars = ['conflict', 'fatalities']
        veg_index_vars = ['NDVI', 'EVI']
        land_use_classes = ['bare', 'built', 'crops', 'flooded_vegetation', 
                            'grass', 'shrub_and_scrub', 'trees', 'water']

        dynamic_col_list = []
        dynamic_col_list += [f"{var}_z_score" for var in climate_vars]
        dynamic_col_list += temp_ns_cols
        dynamic_col_list += precip_cols
        dynamic_col_list += [f"{var}_{radius}" for var in conflict_vars for radius in ['100km', '500km']]
        dynamic_col_list += [f"{var}_12m_{radius}" for var in conflict_vars for radius in ['100km', '500km']]
        dynamic_col_list += [f"{var}_mean" for var in veg_index_vars]
        dynamic_col_list += [f"{var}_z_score" for var in veg_index_vars]
        dynamic_col_list += [f"{var}_z_score_{cls}" for var in veg_index_vars for cls in ['crops', 'grass', 'shrub_and_scrub', 'trees']]
        dynamic_col_list += land_use_classes

        print(f"Total dynamic variables for lagging: {len(dynamic_col_list)}")

        # Write lags directly into DataFrame — avoids holding 600+ Series in memory at once
        # FIX 4: a missing dynamic variable means no lags are built for it and the
        # feature disappears from every model. Collect and fail loudly at the end.
        missing_dynamic = []
        for var in dynamic_col_list:
            if var in self.merged_data.columns:
                grouped = self.merged_data.groupby(self.polygon_id_col)[var]
                for lag in range(1, max_lag + 1):
                    self.merged_data[f'{var}_lag_{lag}'] = grouped.shift(lag)
            else:
                missing_dynamic.append(var)
                print(f"Warning: Variable '{var}' not found in dataset, skipping lag generation.")

        # Lagged deltas for land use classes
        for cls in land_use_classes:
            if cls in self.merged_data.columns:
                grouped = self.merged_data.groupby(self.polygon_id_col)[cls]
                for delta in [1, 2, 3, 6]:
                    self.merged_data[f'{cls}_delta_lag_{delta}'] = grouped.diff(delta).shift(1)
            else:
                missing_dynamic.append(cls)
                print(f"Warning: Land use class '{cls}' not found in dataset, skipping delta computation.")

        if missing_dynamic:
            raise RuntimeError(
                f"{len(missing_dynamic)} dynamic variables are absent from merged_data, "
                f"so no lags were built for them: {missing_dynamic}"
            )

        self.merged_data = self.merged_data.copy()
        print(f"Lag and delta generation completed. Dataset now has {self.merged_data.shape[1]} columns.")


    def generate_seasonal_features_and_lags(self, season1, season2):
        """
        Generates seasonal statistics and 1-year lags for climate, vegetation, and land use variables.
        """
        print("Generating seasonal features and lags...")

        climate_vars = ['avg_temp']
        temp_ns_cols = ['hot_days_ns', 'cold_days_ns',
                        'consec_hot_days_ns', 'consec_cold_days_ns']
        precip_cols = ['spi_1', 'spi_3', 'wet_days_ns', 'consec_dry_days_ns']
        veg_index_vars = ['NDVI', 'EVI']
        land_use_classes = ['bare', 'built', 'crops', 'flooded_vegetation',
                            'grass', 'shrub_and_scrub', 'trees', 'water']

        seasonal_variables = []
        seasonal_variables += [f"{var}_z_score" for var in climate_vars]
        seasonal_variables += temp_ns_cols
        seasonal_variables += precip_cols
        seasonal_variables += [f"{var}_mean" for var in veg_index_vars]
        seasonal_variables += [f"{var}_z_score" for var in veg_index_vars]
        seasonal_variables += [f"{var}_z_score_{cls}" for var in veg_index_vars for cls in ['crops', 'grass', 'shrub_and_scrub', 'trees']]
        seasonal_variables += land_use_classes

        id_col = self.polygon_id_col

        # Only copy the columns needed for seasonal aggregation
        needed_cols = [id_col, 'year', 'month'] + [v for v in seasonal_variables if v in self.merged_data.columns]
        df = self.merged_data[needed_cols].copy()

        def process_season(df, months, season_name):
            df_season = df[df['month'].isin(months)].copy()
            df_season['season_year'] = df_season['year']
            existing_vars = [v for v in seasonal_variables if v in df_season.columns]
            seasonal_agg = df_season.groupby([id_col, 'season_year'])[existing_vars].agg(['max', 'sum', 'mean'])
            seasonal_agg.columns = [f"{var}_{season_name}_{stat}_prev" for var, stat in seasonal_agg.columns]
            return seasonal_agg.reset_index()

        season1_df = process_season(df, season1, 'season1')
        season2_df = process_season(df, season2, 'season2')

        df['effective_season2_year'] = df['year'] - 1
        df['effective_season1_year'] = df['year']
        df.loc[df['month'] < 6, 'effective_season1_year'] -= 1

        df = df.merge(
            season1_df.rename(columns={'season_year': 'effective_season1_year'}),
            on=[id_col, 'effective_season1_year'],
            how='left'
        )

        del season1_df
        gc.collect()

        df = df.merge(
            season2_df.rename(columns={'season_year': 'effective_season2_year'}),
            on=[id_col, 'effective_season2_year'],
            how='left'
        )

        del season2_df
        gc.collect()

        for season in ['season1', 'season2']:
            for var in seasonal_variables:
                for stat in ['max_prev', 'avg_prev']:
                    col = f"{var}_{season}_{stat}_prev"
                    lag_col = f"{var}_{season}_lag"
                    if col in df.columns:
                        df[lag_col] = df.groupby(id_col)[col].shift(1)

        # FIX 3: only the columns CREATED here may be merged back. `needed_cols`
        # holds base variables borrowed from merged_data, which still exist there;
        # merging them back collides on the join and pandas renames BOTH copies to
        # _x/_y. generate_lags_for_dynamic_variables then looks up the bare names,
        # finds nothing, and silently builds no lags for any of them.
        borrowed = set(needed_cols) | {'effective_season1_year', 'effective_season2_year'}
        seasonal_cols = [c for c in df.columns if c not in borrowed]
        self.merged_data = self.merged_data.merge(
            df[[id_col, 'year', 'month'] + seasonal_cols],
            on=[id_col, 'year', 'month'],
            how='left'
        )

        # Guard. NOTE: 'zscore_baseline' is a pre-existing, harmless collision -
        # both merge_temperature_data and merge_precipitation_data carry that
        # metadata label into merged_data, so zscore_baseline_x/_y already exist
        # in every build. It is excluded here so this guard cannot fire spuriously.
        _known_benign = {'zscore_baseline_x', 'zscore_baseline_y'}
        collided = [c for c in self.merged_data.columns
                    if c.endswith(('_x', '_y')) and c not in _known_benign]
        if collided:
            raise RuntimeError(
                f"Seasonal merge produced {len(collided)} unexpected suffixed "
                f"columns: {collided[:10]}"
            )

        del df
        gc.collect()

        print("✅ Seasonal feature generation complete.")


    def generate_wasting_lags(self, last_data_date='2025-03-01'):
        """
        Adds 12-month lags for wasting-related variables.
        """
        print(f"Generating wasting lags and lagged deltas with missing values allowed after {last_data_date}...")

        last_data_date = pd.to_datetime(last_data_date)

        self.merged_data = self.merged_data.sort_values(by=['Ward', 'year', 'month']).reset_index(drop=True)

        if 'date' not in self.merged_data.columns:
            self.merged_data['date'] = pd.to_datetime(self.merged_data[['year', 'month']].assign(day=1))

        # Drop stale future rows for wards that stopped reporting: NaN wasting,
        # dated at or before the last real collection date. Gap rows inserted by
        # reindex_complete_panel are exempt - they must survive so that shift()
        # stays time-aligned.
        _gap = self.merged_data.get(
            'is_gap_row',
            pd.Series(False, index=self.merged_data.index)).fillna(False)
        condition = ~((self.merged_data['wasting'].isna())
                      & (self.merged_data['date'] <= last_data_date)
                      & (~_gap))
        self.merged_data = self.merged_data[condition].reset_index(drop=True)

        for lag in range(1, 13):
            self.merged_data[f'wasting_lag_{lag}'] = self.merged_data.groupby('Ward')['wasting'].shift(lag)
            self.merged_data[f'wasting_obs_{lag}'] = self.merged_data.groupby('Ward')['wasting_count'].shift(lag)
            self.merged_data[f'wasting_risk_lag_{lag}'] = self.merged_data.groupby('Ward')['wasting_risk'].shift(lag)

        for delta in [1, 2, 3, 6]:
            self.merged_data[f'wasting_delta_lag_{delta}'] = (
                self.merged_data.groupby('Ward')['wasting'].diff(delta).shift(1)
            )
            self.merged_data[f'wasting_risk_delta_lag_{delta}'] = (
                self.merged_data.groupby('Ward')['wasting_risk'].diff(delta).shift(1)
            )

        print(f"Lagged wasting variables added. Final shape: {self.merged_data.shape}")
        print("Missing values summary:")
        print(self.merged_data.filter(regex='wasting|risk').isna().sum())


    def save_final_dataset(self, output_path,
                                filename="filtered_ward_level_dataset_2021_01_to_2025_03.pkl"):
        """
        Saves the processed dataset to a pickle file.
        """
        full_path = os.path.join(output_path, filename)
        self.merged_data.to_pickle(full_path)
        print(f"Dataset saved to: {full_path}")


    def plot_seasonal_variable(self, var_name, season='season2', 
                                output_folder="covariates_graphs/seasonal_graphs",
                                data_span='2021_01_to_2025_03'):
        """
        Plots the average value of a specified seasonal variable over time.
        """
        assert season in ['season2', 'season1'], "season must be 'season2' or 'season1'"
        col_name = f"{var_name}_{season}_mean_prev"

        if col_name not in self.merged_data.columns:
            print(f"Column {col_name} not found in dataset.")
            return

        average = self.merged_data.groupby(['year', 'month'])[col_name].mean().reset_index()
        average['plot_date'] = pd.to_datetime(average[['year', 'month']].assign(day=1))

        plt.figure(figsize=(12, 6))
        plt.plot(average['plot_date'], average[col_name], marker='o', label=col_name)
        plt.xlabel('Date')
        plt.ylabel(f'Average {col_name.replace("_", " ").title()}')
        plt.grid(True)
        plt.tight_layout()
        plt.legend()

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            safe_name = col_name.replace(" ", "_")
            file_path = os.path.join(output_folder, f"{safe_name}_{data_span}.png")
            plt.savefig(file_path)
            print(f"Plot saved to: {file_path}")
            plt.close()
        else:
           plt.show()
           plt.close()


#==============================================================
# Initialize
polygon_id_col = "Ward"

builder = WastingPrevalenceDatasetBuilder(
    input_path=INPUT,
    wards_shapefile=wards_shapefile,
    valid_counties=valid_counties,
    historic_muac_data=muac_data,
    new_muac_data=new_muac_data,
    polygon_id_col=polygon_id_col
)

# Run steps
builder.join_datasets_and_eliminate_duplicates()
builder.eliminate_prevalences_with_insufficient_obs(min_num_obs_ward=35)
builder.clean_for_data_continuity(total_months_min=6, max_gap_size=3)
builder.reindex_complete_panel()
print("✅ panel reindex done")

# Merge static & dynamic variables
builder.merge_travel_time()
print("✅ travel_time done")

builder.merge_population_density(
    population_file_name="population_density_2005_2020.csv", map_year=2020)
print("✅ population_density done")

builder.merge_temperature_data(main_file="era5_temperature_stats_2004_01_Onwards.pkl",
                            new_file=f"era5_stats_{start_Month_Year}_to_{end_Month_Year}.pkl",
                            # new_file=None,                           
                            baseline_years=(2004, 2020),
                            analysis_years=(2021, 2026))
print("✅ era5_temperature done")

builder.merge_precipitation_data(main_file="chirps_precipitation_stats_2004_01_Onwards.pkl",
                                new_file=f"chirps_stats_{start_Month_Year}_to_{end_Month_Year}.pkl",
                                # new_file=None,                                
                                baseline_years=(2004, 2020),
                                analysis_years=(2021, 2026))
print("✅ chirps_precipitation done")

builder.merge_ndvi_evi_data(main_file="modis_ndvi_evi_stats_2016_01_Onwards.pkl",
                            new_file=f"modis_ndvi_evi_stats_{start_Month_Year}_to_{end_Month_Year}.pkl",
                            # new_file=None,
                            baseline_years=(2016, 2020),
                            analysis_years=(2021, 2026))
print("✅ ndvi_evi done")

builder.merge_ndvi_evi_by_land_use(main_file="modis_evi_by_land_use_stats_2016_01_Onwards.pkl",
                            new_file=f"modis_evi_by_land_use_stats_{start_Month_Year}_to_{end_Month_Year}.pkl",
                            # new_file=None,                            
                            baseline_years=(2016, 2020),
                            analysis_years=(2021, 2026))
print("✅ ndvi_evi_by_land_use done")

builder.merge_land_use_data(main_file="land_use_pct_stats_2016_01_Onwards.pkl",
                            new_file=f"land_use_pct_stats_{start_Month_Year}_to_{end_Month_Year}.pkl")
                            # new_file=None)
print("✅ land_use done")

builder.merge_conflict_data(conflict_100km_old="ACLED_conflict_12m_100kmcutoff_2021_08_Onwards.pkl",
                            conflict_100km_new=f"ACLED_conflict_12m_running_sum_by_polygon_100kmcutoff_{end_Month_Year1}.pkl",
                            # conflict_100km_new=None,
                            conflict_500km_old="ACLED_conflict_12m_500kmcutoff_2021_08_Onwards.pkl",
                            conflict_500km_new=f"ACLED_conflict_12m_running_sum_by_polygon_500kmcutoff_{end_Month_Year1}.pkl",
                            # conflict_500km_new=None,
                            filter_from_year=2021)
print("✅ conflict done")

# FIX 5: freshness check. When a source file is absent the merge functions print
# one line and proceed with old data, so a broken ingest (expired billing, failed
# GEE export, name mismatch) yields a complete-looking dataset whose geospatial
# columns simply stop months early. Verify coverage before building features.
#
# For a deliberate backfill where a gap is expected and acceptable, raise
# MAX_GEO_LAG_MONTHS rather than removing the check.
MAX_GEO_LAG_MONTHS = 1

_coverage_probes = {
    "ERA5 temperature":      'avg_temp_z_score',
    "CHIRPS precipitation":  'spi_1',
    "MODIS NDVI/EVI":        'NDVI_mean',
    "MODIS EVI by land use": 'NDVI_z_score_grass',
    "Land use percentage":   'crops',
    "ACLED conflict":        'conflict_100km',
}

_md = builder.merged_data
_md_dates = pd.to_datetime(dict(year=_md['year'], month=_md['month'], day=1),
                           errors='coerce')
_reference = _md_dates[_md['wasting'].notna()].max()

print("\n" + "=" * 70)
print(f"Source coverage check (MUAC observed through {_reference:%Y-%m})")
print("=" * 70)
_stale = []
for _label, _col in _coverage_probes.items():
    if _col not in _md.columns:
        print(f"  [MISSING] {_label:24s} column '{_col}' absent")
        _stale.append(f"{_label} (column '{_col}' absent)")
        continue
    _last = _md_dates[_md[_col].notna()].max()
    if pd.isna(_last):
        print(f"  [EMPTY  ] {_label:24s} '{_col}' is entirely null")
        _stale.append(f"{_label} ('{_col}' entirely null)")
        continue
    _gap = (_reference.year - _last.year) * 12 + (_reference.month - _last.month)
    _flag = "OK " if _gap <= MAX_GEO_LAG_MONTHS else "STALE"
    print(f"  [{_flag:7s}] {_label:24s} through {_last:%Y-%m}  ({_gap} month gap)")
    if _gap > MAX_GEO_LAG_MONTHS:
        _stale.append(f"{_label} (through {_last:%Y-%m}, {_gap} month gap)")
print("=" * 70 + "\n")

if _stale:
    raise RuntimeError(
        "Source data is stale or missing, so recent months would carry null "
        "covariates:\n  - " + "\n  - ".join(_stale) +
        f"\nRerun script 2 for the missing period, or raise MAX_GEO_LAG_MONTHS "
        f"(currently {MAX_GEO_LAG_MONTHS}) if this gap is intentional."
    )

# Generate features
builder.extend_future_muac_periods(months_forward=3)
print("✅ extend done")

builder.generate_seasonal_features_and_lags(season1=[3,4,5], season2=[10,11,12])
print("✅ seasonal done")

builder.generate_lags_for_dynamic_variables(max_lag=12)
print("✅ lags done")

temp_muac = new_muac_data.rename(columns={'month_num': 'month', 'Year': 'year'})
temp_muac['date'] = pd.to_datetime(
    temp_muac[['year', 'month']].assign(day=1), 
    errors='coerce'
)
last_muac_date = temp_muac[temp_muac['wasting'].notna()]['date'].max().strftime("%Y-%m-%d")

builder.generate_wasting_lags(last_data_date=last_muac_date)
print("✅ wasting_lags done")

# Save final dataset
builder.merged_data['date'] = pd.to_datetime(
    builder.merged_data[['year', 'month']].assign(day=1),
    errors='coerce'
)
valid_data = builder.merged_data[builder.merged_data['obs_per_month'].notna()]
last_data_date = valid_data['date'].max()
cutoff_date = last_data_date + pd.DateOffset(months=3)

builder.merged_data = builder.merged_data[builder.merged_data['date'] <= cutoff_date].copy()

date_min = builder.merged_data['date'].min()
date_max = builder.merged_data['date'].max()
start_label = date_min.strftime('%b_%Y')
end_label = date_max.strftime('%b_%Y')
update_label = last_data_date.strftime('%b_%Y')

if start_label == end_label:
    date_label = start_label
else:
    date_label = f"{start_label}_to_{end_label}"

builder.save_final_dataset(
    output_path=INPUT,
    filename=f'complete_ward_level_dataset_{date_label}_updated_on_{update_label}.pkl'
)

builder.save_final_dataset(
    output_path=INPUT,
    filename=f'complete_ward_level_dataset.pkl'
)

builder.plot_seasonal_variable(var_name="EVI_z_score_grass", season="season2")