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

#==============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
INPUT = os.path.join(PARENT_DIR, "intermediary_datasets")
SHAPE = os.path.join(PARENT_DIR, "shapefiles")
# INPUT = "/home/ebenezer/Desktop/NDMADEWS_ML_DS/dews-flask-application/Kenya_MUAC_NDMA_implementation/intermediary_datasets"
# SHAPE = "/home/ebenezer/Desktop/NDMADEWS_ML_DS/dews-flask-application/Kenya_MUAC_NDMA_implementation/shapefiles"

# Change directory to the general folder that contains intermediary_datasets folder
os.chdir(PARENT_DIR)
# os.chdir('/home/ebenezer/Desktop/NDMADEWS_ML_DS/dews-flask-application/Kenya_MUAC_NDMA_implementation') 

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
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

start_Month_Year = f"{new_muac_data['Year'].min()}_{str(new_muac_data['month_num'].min()).zfill(2)}"
end_Month_Year = f"{new_muac_data['Year'].max()}_{str(new_muac_data['month_num'].max()).zfill(2)}"

start_Month_Year1 = f"{calendar.month_abbr[(new_muac_data['month_num'].min())].zfill(2)}_{new_muac_data['Year'].min()}"
end_Month_Year1 = f"{calendar.month_abbr[(new_muac_data['month_num'].max())].zfill(2)}_{new_muac_data['Year'].max()}"

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

            # Work at month resolution
            months = missing_idx.to_period('M').sort_values()

            max_run = 1
            current_run = 1

            for prev, curr in zip(months[:-1], months[1:]):
                if curr - prev == 1:  # next month
                    current_run += 1
                    if current_run > max_run:
                        max_run = current_run
                else:
                    current_run = 1

            return max_run        

    def join_datasets_and_eliminate_duplicates(self):
        print("Joining MUAC datasets and eliminating duplicates...")

        # Rename columns for consistency
        self.muac_data = self.muac_data.rename(columns={'month_num': 'month', 'Year': 'year'})
        self.new_muac_data = self.new_muac_data.rename(columns={'month_num': 'month', 'Year': 'year'})

        # Concatenate datasets
        self.muac_data = pd.concat([self.muac_data, self.new_muac_data], ignore_index=True)

        # Check for duplicates
        dupes = self.muac_data[self.muac_data.duplicated(subset=['Ward', 'year', 'month'], keep=False)]
        if not dupes.empty:
            print("Found duplicates:")
            print(dupes.sort_values(['Ward', 'year', 'month'])[['Ward', 'year', 'month', 'wasting']])
        else:
            print("No duplicates found.")

        # Drop duplicates
        self.muac_data = self.muac_data.drop_duplicates(subset=['Ward', 'year', 'month'], keep='last')

        # Print first 5 rows for confirmation
        print(self.muac_data.head())

        # Count observations per month
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
        Optionally plots the eliminated vs. remaining wards.
        
        Parameters:
        - min_num_obs_ward (int): Minimum number of observations required per ward.
        - plot (bool): Whether to display the plot of eliminated wards.
        """
        # Filter the data
        self.muac_data_filtered = self.muac_data[self.muac_data['obs_per_month'] >= min_num_obs_ward]
        
        print(f"Original number of wards: {self.muac_data.Ward.nunique()}, "
            f"Wards left after filtering: {self.muac_data_filtered.Ward.nunique()}")

        # Identify dropped wards
        missing_wards = self.muac_data[~self.muac_data['Ward'].isin(
            self.muac_data_filtered['Ward'])].drop_duplicates(subset=['Ward'])

        # Prepare GeoDataFrames
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

            # Add legend
            missing_patch = mpatches.Patch(color='red', label='Eliminated Wards')
            filtered_patch = mpatches.Patch(color='lightblue', label='Remaining Wards')
            plt.legend(handles=[missing_patch, filtered_patch], loc='lower left', fontsize='x-small')

            plt.title("Wards After Minimum Observations Filtering")
            plt.tight_layout()
#            plt.show()
#            plt.close()

    
    def clean_for_data_continuity(self, total_months_min=6, max_gap_size=3, 
                                    folder_images="plots", save_plots=True, plot=False):
            """
            Cleans MUAC data for continuity. Filters out wards with too few total months,
            large data gaps, or discontinuous data. Optionally generates plots of
            inconsistently sampled wards and summary figures.
            """
    
            print("Starting data continuity cleaning...")
    
            # Ensure year and month are integers
            self.muac_data_filtered['year'] = self.muac_data_filtered['year'].astype(int)
            self.muac_data_filtered['month'] = self.muac_data_filtered['month'].astype(int)
    
            # Ensure date column exists with strict error handling
            self.muac_data_filtered['date'] = pd.to_datetime(
                self.muac_data_filtered[['year', 'month']].assign(day=1),
                errors='raise'
            )
    
            # Summary of each ward's date coverage
            ward_summary = (
                self.muac_data_filtered.groupby('Ward')
                .agg(
                    first_observation=('date', 'min'),
                    last_observation=('date', 'max'),
                    total_months=('date', 'nunique')
                )
                .reset_index()
            )
    
            def calculate_gaps(row):
    
                # Work only with this ward
                ward_data = self.muac_data_filtered[
                    self.muac_data_filtered['Ward'] == row['Ward']
                ].dropna(subset=['wasting'])
    
                # Build observed dates from year + month (again, independent of any old columns)
                ward_data = ward_data.sort_values(['year', 'month'])
                observed_dates = pd.DatetimeIndex(
                    pd.to_datetime(ward_data[['year','month']].assign(day=1))
                ).sort_values().unique()

    
                # Global dataset start/end (from filtered data)
                dataset_start = self.muac_data_filtered['date'].min()
                dataset_end = self.muac_data_filtered['date'].max()
    
                # Full range between first and last observation for this ward
                full_range = pd.date_range(start=row['first_observation'],
                                        end=row['last_observation'], freq='MS')
                missing_months = full_range.difference(observed_dates)
    
                #  use longest continuous gap, not total missing months
                gap_size = self._max_consecutive_gap(missing_months)
    
    
                # Initial gap before first observation
                initial_gap_range = pd.date_range(
                    start=dataset_start,
                    end=row['first_observation'] - pd.DateOffset(months=1),
                    freq='MS'
                )
                initial_gap_size = len(initial_gap_range.difference(observed_dates))
    
                # End gap after last observation
                end_gap_range = pd.date_range(
                    start=row['last_observation'] + pd.DateOffset(months=1),
                    end=dataset_end,
                    freq='MS'
                )
                end_gap_size = len(end_gap_range.difference(observed_dates))
    
                # Pre-2024 gaps
                jan_2024 = pd.Timestamp('2024-01-01')
                pre_2024_range = pd.date_range(
                    start=row['first_observation'],
                    end=jan_2024 - pd.DateOffset(months=1),
                    freq='MS'
                )
                pre_2024_gaps = pre_2024_range.difference(observed_dates)
                pre_2024_gap_size = len(pre_2024_gaps)
    
                # Post-2024 continuity
                ward_time = pd.DatetimeIndex(pd.to_datetime(
                    ward_data[['year','month']].assign(day=1))).sort_values().unique()
                post_2024_time = ward_time[ward_time >= jan_2024]  # DatetimeIndex

                
                if post_2024_time.size == 0:
                    post_2024_continuous = False
                else:
                    post_2024_range = pd.date_range(
                        start=jan_2024,
                        end=row['last_observation'],
                        freq='MS'
                    )
                    post_2024_gaps = post_2024_range.difference(post_2024_time)
                    post_2024_continuous = (len(post_2024_gaps) == 0)
                
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
    
            # --- Compute gaps + build ward summary (ONCE) ---
            gap_details = ward_summary.apply(calculate_gaps, axis=1)
            final_ward_summary = pd.concat([ward_summary, gap_details], axis=1)
            
            print(final_ward_summary[['Ward', 'first_observation', 'last_observation',
                                      'total_months', 'gap_size', 'initial_gap_size',
                                      'end_gap_size', 'pre_2024_gap_size',
                                      'post_2024_continuous', 'continuity_status']])
            
            # --- Apply selection rules (same as yours) ---
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
            
            # IMPORTANT: .copy() avoids chained-assignment weirdness later
            self.muac_data_filtered = self.muac_data_filtered[
                (self.muac_data_filtered['Ward'].isin(valid_wards_list)) |
                ((self.muac_data_filtered['Ward'].isin(post_2024_only_list)) &
                 (self.muac_data_filtered['year'] >= 2024))
            ].copy()
            
            # Rebuild date from year+month (same as yours)
            self.muac_data_filtered['date'] = pd.to_datetime(
                self.muac_data_filtered[['year', 'month']].assign(day=1),
                errors='raise'
            )
            
            start_date = self.muac_data_filtered['date'].min().strftime('%Y_%m')
            end_date   = self.muac_data_filtered['date'].max().strftime('%Y_%m')
            print(f"MUAC data time frame: {start_date} to {end_date}")
            
            # --- Keep HIS file naming (two outputs) ---
            filename  = f"Kenya_NDMA_MUAC_ward_level_{start_date}_to_{end_date}.pkl"
            filename2 = f"Kenya_NDMA_MUAC_ward_level_{start_date}_Onwards.pkl"
            
            output_path1 = os.path.join(self.input_path, filename)
            output_path2 = os.path.join(self.input_path, filename2)
            
            self.muac_data_filtered.to_pickle(output_path1)
            self.muac_data_filtered.to_pickle(output_path2)
            
            print(f"Filtered MUAC dataset saved to {output_path2}")  # prints the last one like his
            
            print(f"Total selected wards: {len(valid_wards_list) + len(post_2024_only_list)}")
            print(f"Wards selected due to post-2024 continuity: {len(post_2024_only_list)}")
            print(f"Final dataset shape: {self.muac_data_filtered.shape}")
            
            # --- Folders for plots ---
            inconsistent_folder = os.path.join(folder_images, "inconsistent_wards")
            summary_folder = os.path.join(folder_images, "summary_plots")
            if save_plots:
                os.makedirs(inconsistent_folder, exist_ok=True)
                os.makedirs(summary_folder, exist_ok=True)
            
            # === 1) Plot per post-2024-only ward (same idea as yours)
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
            
            # === 2) Plot average wasting prevalence (all vs filtered)
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
            
            # === 3) Map of eliminated vs remaining wards
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



    def merge_travel_time(self, generate_plot=True, save_plot=True, plot=False, 
                          output_dir="covariates_graphs", accessiblity_file_name="accessibility_to_cities_2015.csv"):
        """
        Merges travel time data and optionally generates a heatmap.
        """
        print("Merging travel time data...")
        travel_time = pd.read_csv(os.path.join(self.input_path, accessiblity_file_name))
        self.merged_data = pd.merge(self.muac_data_filtered, travel_time, 
                                    on=self.polygon_id_col, how='left')

        if not generate_plot:
            return  # Skip plotting entirely

        self.wards_shapefile_final = self.wards_shapefile[self.wards_shapefile[self.polygon_id_col].isin(self.merged_data[self.polygon_id_col])].copy()
        self.wards_shapefile_final = self.wards_shapefile_final.merge(
            self.merged_data[[self.polygon_id_col, 'travel_time_to_cities_2015']], on=self.polygon_id_col, how='left'
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

        if not generate_plot:
            return  # Skip plotting entirely

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

        # Load datasets
        old_path = os.path.join(self.input_path, main_file)
        old_data = pd.read_pickle(old_path)

        # Try loading new data
        # new_path = os.path.join(self.input_path, new_file)
        # if os.path.exists(new_path):

        if new_file is not None:
            new_path = os.path.join(self.input_path, new_file)
            if os.path.exists(new_path):


                try:
                    new_data = pd.read_pickle(new_path)
                    full_data = pd.concat([old_data, new_data], ignore_index=True)
                    print(f"New ERA5 data found and merged with old data.")

                        # Get full date range from merged data
                    if 'year' in full_data.columns and 'month' in full_data.columns:
                        full_data['date'] = pd.to_datetime(
                            full_data['year'].astype(str) + '-' + full_data['month'].astype(str) + '-01')
                        start_date = full_data['date'].min().strftime('%Y_%m')
                        end_date = full_data['date'].max().strftime('%Y_%m')
                    else:
                        raise ValueError("Columns 'year' and 'month' must be present in the data.")

                    # Build new filename
                    new_filename = f"era5_temperature_stats_{start_date}_Onwards.pkl"

                    # Save
                    output_path = os.path.join(self.input_path, new_filename)
                    full_data.to_pickle(output_path)
                    print(f"Saved merged dataset to '{output_path}'.")
                        # Delete old and new files
                    # try:
                    #     os.remove(old_path)
                    #     print(f"Deleted old file: {old_path}")
                    # except Exception as e:
                    #     print(f"Could not delete old file: {e}")

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

        # Remove duplicates
        dupes = full_data.duplicated(subset=[self.polygon_id_col, 
                                             'year', 'month'], keep=False)
        if dupes.any():
            print("Duplicates found in temperature data:")
            print(full_data.loc[dupes, [self.polygon_id_col, 'year', 'month', 'avg_temp_month']])
            full_data = full_data.drop_duplicates(
                subset=[self.polygon_id_col, 'year', 'month'], keep='last')

        # Prepare and sort by date
        full_data['date'] = pd.to_datetime(full_data[['year', 'month']].assign(day=1))
        full_data = full_data.sort_values('date').copy()

        # Baseline and analysis windows
        baseline = full_data[(
            full_data['year'] >= baseline_years[0]) & (full_data['year'] <= baseline_years[1])]

        analysis = full_data[(full_data['year'] >= analysis_years[0])].copy()

        # Monthly baseline stats
        monthly_stats = baseline.groupby('month').agg(
            avg_temp_longterm=('avg_temp_month', 'mean'),
            avg_temp_std=('avg_temp_month', 'std'),
            hot_days_longterm=('hot_days', 'mean'),
            hot_days_std=('hot_days', 'std'),
            cold_days_longterm=('cold_days', 'mean'),
            cold_days_std=('cold_days', 'std'),
            consec_hot_days_longterm=('consec_hot_days', 'mean'),
            consec_hot_days_std=('consec_hot_days', 'std'),
            consec_cold_days_longterm=('consec_cold_days', 'mean'),
            consec_cold_days_std=('consec_cold_days', 'std')
        ).reset_index()

        # Merge baseline stats into analysis
        analysis = pd.merge(analysis, monthly_stats, on='month', how='left')

        # Compute Z-scores
        analysis['avg_temp_z_score'] = (analysis['avg_temp_month'] - analysis['avg_temp_longterm']) / analysis['avg_temp_std']
        analysis['hot_days_z_score'] = (analysis['hot_days'] - analysis['hot_days_longterm']) / analysis['hot_days_std']
        analysis['cold_days_z_score'] = (analysis['cold_days'] - analysis['cold_days_longterm']) / analysis['cold_days_std']
        analysis['consec_hot_days_z_score'] = (analysis['consec_hot_days'] - analysis['consec_hot_days_longterm']) / analysis['consec_hot_days_std']
        analysis['consec_cold_days_z_score'] = (analysis['consec_cold_days'] - analysis['consec_cold_days_longterm']) / analysis['consec_cold_days_std']

        analysis['zscore_baseline'] = f"{baseline_years[0]}–{baseline_years[1]}"

        # Select relevant columns
        filtered_temp_data = analysis[[self.polygon_id_col, 'year', 'month',
                                        'avg_temp_month', 'avg_temp_z_score',
                                        'hot_days', 'hot_days_z_score',
                                        'cold_days', 'cold_days_z_score',
                                        'consec_hot_days', 'consec_hot_days_z_score',
                                        'consec_cold_days', 'consec_cold_days_z_score',
                                        'zscore_baseline']]

        # Merge temperature data
        expected_polygons = self.merged_data[self.polygon_id_col].unique()
        filtered_temp_data = filtered_temp_data[
            filtered_temp_data[self.polygon_id_col].isin(expected_polygons)
        ]
        self.merged_data = pd.merge(self.merged_data, filtered_temp_data,
                                    on=[self.polygon_id_col, 'year', 'month'], how='left')
        print("Temperature data merged successfully.")

        # === Plotting Section ===
        if not generate_plot:
            return

        avg_temp = (
            filtered_temp_data.groupby(['year', 'month'])
            .agg(average_temp=('avg_temp_z_score', 'mean'))
            .reset_index()
        )
        avg_temp['date'] = pd.to_datetime(avg_temp[['year', 'month']].assign(day=1))

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
                                analysis_years=(2021, 2025),
                                generate_plot=True, save_plot=True, plot=False,
                                output_dir="covariates_graphs"):
        """
        Merges CHIRPS precipitation data if new file exists, computes Z-scores,
        and optionally generates plots including enhanced precipitation vs wasting overlay.
        """
        print("Merging CHIRPS precipitation data...")

        # Load old data
        old_path = os.path.join(self.input_path, main_file)
        old_data = pd.read_pickle(old_path)

        # Try loading new data
        if new_file is not None:
            new_path = os.path.join(self.input_path, new_file)
            if os.path.exists(new_path):
                try:
                    new_data = pd.read_pickle(new_path)
                    full_data = pd.concat([old_data, new_data], ignore_index=True)
                    print(f"New CHIRPS data found and merged with old data.")

                    if 'year' in full_data.columns and 'month' in full_data.columns:
                        full_data['date'] = pd.to_datetime(
                            full_data['year'].astype(str) + '-' + full_data['month'].astype(str) + '-01')
                        start_date = full_data['date'].min().strftime('%Y_%m')
                        end_date = full_data['date'].max().strftime('%Y_%m')
                    else:
                        raise ValueError("Columns 'year' and 'month' must be present in the data.")

                    # Build new filename
                    new_filename = f"chirps_precipitation_stats_{start_date}_Onwards.pkl"

                    # Save
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

        # Handle duplicates
        dupes = full_data.duplicated(subset=[self.polygon_id_col, 'year', 'month'], keep=False)
        if dupes.any():
            print("Duplicates found in precipitation data:")
            print(full_data.loc[dupes, [self.polygon_id_col, 'year', 'month', 'precip_total']])
            full_data = full_data.drop_duplicates(subset=[self.polygon_id_col, 'year', 'month'], keep='last')

        # Prepare date column
        full_data['date'] = pd.to_datetime(full_data[['year', 'month']].assign(day=1))
        full_data = full_data.sort_values('date').copy()

        # Baseline and analysis windows
        baseline = full_data[(full_data['year'] >= baseline_years[0]) & (full_data['year'] <= baseline_years[1])]
        analysis = full_data[(full_data['year'] >= analysis_years[0])].copy()

        # Monthly baseline stats
        monthly_stats = baseline.groupby('month').agg(
            total_precip_longterm=('precip_total', 'mean'),
            total_precip_std=('precip_total', 'std'),
            wet_days_longterm=('wet_days', 'mean'),
            wet_days_std=('wet_days', 'std'),
            dry_days_longterm=('dry_days', 'mean'),
            dry_days_std=('dry_days', 'std'),
            consec_wet_days_longterm=('consec_wet_days', 'mean'),
            consec_wet_days_std=('consec_wet_days', 'std'),
            consec_dry_days_longterm=('consec_dry_days', 'mean'),
            consec_dry_days_std=('consec_dry_days', 'std')
        ).reset_index()

        # Merge baseline stats into analysis
        analysis = pd.merge(analysis, monthly_stats, on='month', how='left')

        # Compute Z-scores
        analysis['precip_total_z_score'] = (analysis['precip_total'] - analysis['total_precip_longterm']) / analysis['total_precip_std']
        analysis['wet_days_z_score'] = (analysis['wet_days'] - analysis['wet_days_longterm']) / analysis['wet_days_std']
        analysis['dry_days_z_score'] = (analysis['dry_days'] - analysis['dry_days_longterm']) / analysis['dry_days_std']
        analysis['consec_wet_days_z_score'] = (analysis['consec_wet_days'] - analysis['consec_wet_days_longterm']) / analysis['consec_wet_days_std']
        analysis['consec_dry_days_z_score'] = (analysis['consec_dry_days'] - analysis['consec_dry_days_longterm']) / analysis['consec_dry_days_std']

        analysis['zscore_baseline'] = f"{baseline_years[0]}–{baseline_years[1]}"

        # Select final columns
        filtered_data = analysis[[self.polygon_id_col, 'year', 'month', 
                                    'precip_total', 'precip_total_z_score', 
                                    'wet_days', 'wet_days_z_score', 
                                    'dry_days', 'dry_days_z_score', 
                                    'consec_wet_days', 'consec_wet_days_z_score', 
                                    'consec_dry_days', 'consec_dry_days_z_score',
                                    'zscore_baseline']]

        # Merge into master dataset
        expected_polygons = self.merged_data[self.polygon_id_col].unique()
        filtered_prec_data = filtered_data[
            filtered_data[self.polygon_id_col].isin(expected_polygons)
        ]

        self.merged_data = pd.merge(self.merged_data, 
                                    filtered_prec_data, 
                                    on=[self.polygon_id_col, 'year', 'month'], 
                                    how='left')
        print("✅ Precipitation data merged successfully.")

        # === Plotting Section ===
        if not generate_plot:
            return

        # === PLOT 1: Basic Average Precipitation Z-score ===
        avg_precip = (
            filtered_prec_data.groupby(['year', 'month'])
            .agg(avg_precip=('precip_total_z_score', 'mean'))
            .reset_index()
        )
        avg_precip['date'] = pd.to_datetime(avg_precip[['year', 'month']].assign(day=1))

        plt.figure(figsize=(12, 6))
        plt.plot(avg_precip['date'], avg_precip['avg_precip'], marker='o', color='b', label='Average Precipitation Z-score')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Longterm normal (y=0)')
        plt.xlabel('Date')
        plt.ylabel('Average Total Precipitation Z-score')
        plt.title('Average Precipitation Z-Score Over Time')
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

        # Colors
        C_PRECIP = "#1f77b4"   # precip line (blue)
        C_ZERO   = "#6e6e6e"   # y=0 line (grey)
        C_ANOM   = "#8c2d04"   # anomaly fill (reddish-brown)
        C_LONG   = "#cfe8ff"   # long rains shading (light blue)
        C_SHORT  = "#b2f1e5"   # short rains shading (turquoise)
        C_WASTE  = "#e6550d"   # wasting_count line (orange-red, high contrast)

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # precip z-score (left y-axis)
        ax1.plot(pm["date"], pm["avg_precip"],
                color=C_PRECIP, linewidth=2, marker="o", markersize=4,
                label="Precipitation Z-score")
        ax1.axhline(y=0, linestyle="--", linewidth=1.2, color=C_ZERO, label="Long-term normal (y=0)")

        # negative anomaly shading
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

        # season shading
        xmin, xmax = pm["date"].min(), pm["date"].max()
        years = np.sort(pm["date"].dt.year.unique())
        added_long, added_short = False, False
        for y in years:
            a0, b0 = max(pd.Timestamp(y,3,1), xmin), min(pd.Timestamp(y,6,1), xmax)   # Mar–May
            if a0 < b0:
                ax1.axvspan(a0, b0, color=C_LONG, alpha=0.35,
                            label="Long rains (Mar–May)" if not added_long else None)
                added_long = True
            a1, b1 = max(pd.Timestamp(y,10,1), xmin), min(pd.Timestamp(y+1,1,1), xmax) # Oct–Dec
            if a1 < b1:
                ax1.axvspan(a1, b1, color=C_SHORT, alpha=0.35,
                            label="Short rains (Oct–Dec)" if not added_short else None)
                added_short = True

        # wasting_count as a contrasting LINE on right axis
        ax2 = ax1.twinx()
        ax2.set_ylabel("Counts")
        ax2.plot(pm["date"], pm["wasting_count"],
                color=C_WASTE, linewidth=2.2, marker="s", markersize=4,
                linestyle="-", label="Wasting count")

        # tidy x-axis dates
        fig.autofmt_xdate()

        # legend (ensure anomaly text shows even if no negatives in window)
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
                            analysis_years=(2021, 2025),
                            generate_plot=True, save_plot=True, plot=False,
                            output_dir="covariates_graphs"):
        """
        Merges MODIS NDVI/EVI data, computes Z-scores from fixed baseline, 
        and optionally generates plots including enhanced EVI vs wasting overlay.
        """
        print("Merging MODIS NDVI/EVI data...")

        # Load old data
        old_path = os.path.join(self.input_path, main_file)
        old_data = pd.read_pickle(old_path)

        # Try loading new data
        if new_file is not None:
            new_path = os.path.join(self.input_path, new_file)
            if os.path.exists(new_path):
                try:
                    new_data = pd.read_pickle(new_path)
                    full_data = pd.concat([old_data, new_data], ignore_index=True)
                    print("New NDVI/EVI data found and merged with old data.")

                    if 'year' in full_data.columns and 'month' in full_data.columns:
                        full_data['date'] = pd.to_datetime(
                            full_data['year'].astype(str) + '-' + full_data['month'].astype(str) + '-01')
                        start_date = full_data['date'].min().strftime('%Y_%m')
                        end_date = full_data['date'].max().strftime('%Y_%m')
                    else:
                        raise ValueError("Columns 'year' and 'month' must be present in the data.")

                    # Build new filename
                    new_filename = f"modis_ndvi_evi_stats_{start_date}_Onwards.pkl"

                    # Save
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

        # Handle duplicates
        dupes = full_data.duplicated(subset=[self.polygon_id_col, 'year', 'month'], keep=False)
        if dupes.any():
            print("Duplicates found in NDVI/EVI data:")
            print(full_data.loc[dupes, [self.polygon_id_col, 'year', 'month', 'EVI_mean']])
            full_data = full_data.drop_duplicates(subset=[self.polygon_id_col, 'year', 'month'], keep='last')

        # Prepare date column
        full_data['date'] = pd.to_datetime(full_data[['year', 'month']].assign(day=1))
        full_data = full_data.sort_values('date').copy()

        # Baseline and analysis windows
        baseline = full_data[(full_data['year'] >= baseline_years[0]) & (full_data['year'] <= baseline_years[1])]
        analysis = full_data[(full_data['year'] >= analysis_years[0])].copy()

        # Monthly baseline stats
        monthly_stats = baseline.groupby('month').agg(
            NDVI_longterm=('NDVI_mean', 'mean'),
            NDVI_std=('NDVI_mean', 'std'),
            EVI_longterm=('EVI_mean', 'mean'),
            EVI_std=('EVI_mean', 'std')
        ).reset_index()

        # Merge baseline into analysis
        analysis = pd.merge(analysis, monthly_stats, on='month', how='left')

        # Compute Z-scores
        analysis['NDVI_z_score'] = (analysis['NDVI_mean'] - analysis['NDVI_longterm']) / analysis['NDVI_std']
        analysis['EVI_z_score'] = (analysis['EVI_mean'] - analysis['EVI_longterm']) / analysis['EVI_std']
        analysis['zscore_baseline_EVI'] = f"{baseline_years[0]}–{baseline_years[1]}"

        # Final columns
        filtered_evi_data = analysis[[self.polygon_id_col, 'year', 'month',
                                'NDVI_mean', 'NDVI_z_score',
                                'EVI_mean', 'EVI_z_score',
                                'zscore_baseline_EVI']]

        # Merge into master dataset
        expected_polygons = self.merged_data[self.polygon_id_col].unique()
        filtered_evi_data = filtered_evi_data[
            filtered_evi_data[self.polygon_id_col].isin(expected_polygons)
        ]
        self.merged_data = pd.merge(self.merged_data, 
                                    filtered_evi_data, 
                                    on=[self.polygon_id_col, 'year', 'month'], 
                                    how='left')
        print("✅ NDVI/EVI data merged successfully.")

        # === Plotting Section ===
        if not generate_plot:
            return

        # === PLOT 1: Basic Average NDVI/EVI Z-scores ===
        avg_z = (
            filtered_evi_data.groupby(['year', 'month'])
            .agg(avg_NDVI_z_score=('NDVI_z_score', 'mean'),
                avg_EVI_z_score=('EVI_z_score', 'mean'))
            .reset_index()
        )
        avg_z['date'] = pd.to_datetime(avg_z[['year', 'month']].assign(day=1))

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

        # === PLOT 2: Enhanced EVI/NDVI vs Wasting ===
        if "wasting_count" not in self.merged_data.columns:
            print("[!] Skipping enhanced EVI plot - missing wasting_count column")
            return

        # Monthly wasting_count
        wasting_by_month = (
            self.merged_data.groupby(["year","month"], as_index=False)
            .agg(wasting_count=("wasting_count","sum"))
        )
        wasting_by_month["date"] = pd.to_datetime(wasting_by_month[["year","month"]].assign(day=1))

        # Align on year-month-date
        pm_evi = (
            avg_z.merge(
                wasting_by_month[["year","month","date","wasting_count"]],
                on=["year","month","date"], how="inner"
            )
            .sort_values("date")
        )

        # Colors
        C_EVI   = "#1f77b4"   # EVI line (blue)
        C_ZERO  = "#6e6e6e"   # y=0 line (grey)
        C_ANOM  = "#8c2d04"   # anomaly fill (reddish-brown)
        C_LONG  = "#cfe8ff"   # long rains (light blue)
        C_SHORT = "#b2f1e5"   # short rains (turquoise)
        C_WASTE = "#e6550d"   # wasting_count (orange-red, high contrast)

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # EVI line (left axis)
        ax1.plot(pm_evi["date"], pm_evi["avg_EVI_z_score"],
                color=C_EVI, linewidth=2, marker="o", markersize=4,
                label="Average EVI Z-Score")

        ax1.axhline(y=0, linestyle="--", 
                    linewidth=1.2, color=C_ZERO, label="Long-term normal (y=0)")

        # Shade negative EVI anomalies
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

        # Season bands across all years
        xmin, xmax = pm_evi["date"].min(), pm_evi["date"].max()
        years = np.sort(pm_evi["date"].dt.year.unique())
        added_long = added_short = False
        for y in years:
            # Long rains: Mar–May (to Jun 1)
            a0, b0 = max(pd.Timestamp(y,3,1), xmin), min(pd.Timestamp(y,6,1), xmax)
            if a0 < b0:
                ax1.axvspan(a0, b0, color=C_LONG, alpha=0.35,
                            label="Long rains (Mar–May)" if not added_long else None)
                added_long = True
            # Short rains: Oct–Dec (to Jan 1 next year)
            a1, b1 = max(pd.Timestamp(y,10,1), xmin), min(pd.Timestamp(y+1,1,1), xmax)
            if a1 < b1:
                ax1.axvspan(a1, b1, color=C_SHORT, alpha=0.35,
                            label="Short rains (Oct–Dec)" if not added_short else None)
                added_short = True

        ax1.set_xlabel("Date")
        ax1.set_ylabel("Average EVI Z-Score")

        # Wasting count as contrasting LINE (right axis)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Wasting count")
        ax2.plot(pm_evi["date"], pm_evi["wasting_count"],
                color=C_WASTE, linewidth=2.2, marker="D", markersize=4,
                label="Wasting count")

        # Legend (include anomaly proxy if needed)
        lines1, labels1 = ax1.get_legend_handles_labels()
        if not neg_mask.any():
            lines1.append(anomaly_patch)
            labels1.append(anomaly_patch.get_label())
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        # Finish
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
                                    # new_file="modis_evi_by_land_use_stats_2019_08_to_2025_03.pkl",
                                    new_file=None,
                                    baseline_years=(2016, 2020),
                                    analysis_years=(2021, 2025),
                                    generate_plot=True, save_plot=True, plot=False,
                                    output_dir="covariates_graphs"):
        """
        Merges NDVI/EVI data by land use class, computes Z-scores from a fixed baseline, 
        and optionally plots average EVI Z-scores over time by land use.
        """
        print("Merging NDVI/EVI data by land use class...")

        # Load old data
        old_path = os.path.join(self.input_path, main_file)
        old_data = pd.read_pickle(old_path)

        # Try to load new data
        # new_path = os.path.join(self.input_path, new_file)
        # if os.path.exists(new_path):

        if new_file is not None:
            new_path = os.path.join(self.input_path, new_file)
            if os.path.exists(new_path):

                try:
                    new_data = pd.read_pickle(new_path)
                    full_data = pd.concat([old_data, new_data], ignore_index=True)
                    print("New land use NDVI/EVI data found and merged with old data.")

                    if 'year' in full_data.columns and 'month' in full_data.columns:
                        full_data['date'] = pd.to_datetime(
                            full_data['year'].astype(str) + '-' + full_data['month'].astype(str) + '-01')
                        start_date = full_data['date'].min().strftime('%Y_%m')
                        end_date = full_data['date'].max().strftime('%Y_%m')
                    else:
                        raise ValueError("Columns 'year' and 'month' must be present in the data.")

                    # Build new filename
                    new_filename = f"modis_evi_by_land_use_stats_{start_date}_Onwards.pkl"

                    # Save
                    output_path = os.path.join(self.input_path, new_filename)
                    full_data.to_pickle(output_path)
                    print(f"Saved merged dataset to '{output_path}'.")
                        # Delete old and new files
                    # try:
                    #     os.remove(old_path)
                    #     print(f"Deleted old file: {old_path}")
                    # except Exception as e:
                    #     print(f"Could not delete old file: {e}")

                    try:
                        os.remove(new_path)
                        print(f"Deleted new file: {new_path}")
                    except Exception as e:
                        print(f"Could not delete new file: {e}")


                except Exception as e:
                    print(f"Error loading new file: {e}. Proceeding with old data only.")
                    full_data = old_data
        else:
            print("No new file found. Proceeding with old data only.")
            full_data = old_data

        # Handle duplicates
        dupes = full_data.duplicated(subset=[self.polygon_id_col, 'year', 'month', 'land_use_class'], keep=False)
        if dupes.any():
            print("Duplicates found in land use NDVI/EVI data:")
            print(full_data.loc[dupes, [self.polygon_id_col, 'year', 'month', 'land_use_class', 'EVI']])
            full_data = full_data.drop_duplicates(subset=[self.polygon_id_col, 'year', 'month', 'land_use_class'], keep='last')

        # Create date column
        full_data['date'] = pd.to_datetime(full_data[['year', 'month']].assign(day=1))

        # Baseline and analysis periods
        baseline = full_data[(full_data['year'] >= baseline_years[0]) & (full_data['year'] <= baseline_years[1])]
        analysis = full_data[full_data['year'] >= analysis_years[0]].copy()

        # Compute baseline stats
        longterm_stats = baseline.groupby(['land_use_class', 'month']).agg(
            NDVI_longterm_mean=('NDVI', 'mean'),
            NDVI_longterm_std=('NDVI', 'std'),
            EVI_longterm_mean=('EVI', 'mean'),
            EVI_longterm_std=('EVI', 'std')
        ).reset_index()

        # Ensure full month-class combinations
        all_combinations = pd.MultiIndex.from_product(
            [baseline['land_use_class'].dropna().unique(), range(1, 13)],
            names=['land_use_class', 'month']
        ).to_frame(index=False)
        longterm_stats = pd.merge(all_combinations, longterm_stats, on=['land_use_class', 'month'], how='left')

        # Merge with analysis
        analysis = pd.merge(analysis, longterm_stats, on=['land_use_class', 'month'], how='left')

        # Compute Z-scores
        analysis['NDVI_z_score'] = (analysis['NDVI'] - analysis['NDVI_longterm_mean']) / analysis['NDVI_longterm_std']
        analysis['EVI_z_score'] = (analysis['EVI'] - analysis['EVI_longterm_mean']) / analysis['EVI_longterm_std']
        analysis['zscore_baseline'] = f"{baseline_years[0]}–{baseline_years[1]}"

        # Pivot to wide format
        pivot = analysis.pivot_table(
            index=[self.polygon_id_col, 'year', 'month'],
            columns='land_use_class',
            values=['NDVI', 'EVI', 'NDVI_z_score', 'EVI_z_score']
        )
        pivot.columns = [f"{var}_{cls}" for var, cls in pivot.columns]
        pivot.reset_index(inplace=True)

        # Merge into master dataset
        expected_polygons = self.merged_data[self.polygon_id_col].unique()
        filtered_pivot_data = pivot[
            pivot[self.polygon_id_col].isin(expected_polygons)
        ]

        self.merged_data = pd.merge(self.merged_data, filtered_pivot_data, 
                                    on=[self.polygon_id_col, 'year', 'month'], 
                                    how='left')
        print("✅ NDVI/EVI by land use class merged.")

        # === Plotting Section ===
        if not generate_plot:
            return

        avg_z = filtered_pivot_data.groupby(['year', 'month']).agg(
            avg_EVI_z_score_grass=('EVI_z_score_grass', 'mean'),
            avg_EVI_z_score_shrub_and_scrub=('EVI_z_score_shrub_and_scrub', 'mean'),
            avg_EVI_z_score_trees=('EVI_z_score_trees', 'mean'),
            avg_EVI_z_score_crops=('EVI_z_score_crops', 'mean')
        ).reset_index()
        avg_z['date'] = pd.to_datetime(avg_z[['year', 'month']].assign(day=1))

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
                            # new_file="land_use_pct_stats_2019_08_to_2025_03.pkl",
                            new_file=None,
                            generate_plot=True, save_plot=True, plot=True,
                            output_dir="covariates_graphs"):
        """
        Merges land use percentage data, pivots to wide format,
        and optionally generates plots of average land use percentages over time.
        """
        print("Merging land use data...")

        # Load datasets
        old_path = os.path.join(self.input_path, main_file)
        old_data = pd.read_pickle(old_path)

        if new_file is not None:
            new_path = os.path.join(self.input_path, new_file)
            if os.path.exists(new_path):

        # new_path = os.path.join(self.input_path, new_file)
        # if os.path.exists(new_path):
                try:
                    new_data = pd.read_pickle(new_path)
                    full_data = pd.concat([old_data, new_data], ignore_index=True)
                    print("New land use data found and merged with old data.")


                    if 'year' in full_data.columns and 'month' in full_data.columns:
                        full_data['date'] = pd.to_datetime(
                            full_data['year'].astype(str) + '-' + full_data['month'].astype(str) + '-01')
                        start_date = full_data['date'].min().strftime('%Y_%m')
                        end_date = full_data['date'].max().strftime('%Y_%m')
                    else:
                        raise ValueError("Columns 'year' and 'month' must be present in the data.")

                    # Build new filename
                    new_filename = f"land_use_pct_stats_{start_date}_Onwards.pkl"

                    # Save
                    output_path = os.path.join(self.input_path, new_filename)
                    full_data.to_pickle(output_path)
                    print(f"Saved merged dataset to '{output_path}'.")
                        # Delete old and new files
                    # try:
                    #     os.remove(old_path)
                    #     print(f"Deleted old file: {old_path}")
                    # except Exception as e:
                    #     print(f"Could not delete old file: {e}")

                    try:
                        os.remove(new_path)
                        print(f"Deleted new file: {new_path}")
                    except Exception as e:
                        print(f"Could not delete new file: {e}")


                except Exception as e:
                    print(f"Error loading new file: {e}. Proceeding with old data only.")
                    full_data = old_data
        else:
            print("No new land use file found. Proceeding with old data only.")
            full_data = old_data

        # Handle duplicates
        dupes = full_data.duplicated(
            subset=[self.polygon_id_col, 'year', 'month', 'land_use_name'], keep=False)
        if dupes.any():
            print("Duplicates found in land use data:")
            print(full_data.loc[dupes, [self.polygon_id_col, 'year', 
                                        'month', 'land_use_name', 'land_use_percentage']])
            full_data = full_data.drop_duplicates(
                subset=[self.polygon_id_col, 'year', 'month', 'land_use_name'], keep='last')


        # Ensure proper date
        full_data['month'] = pd.to_numeric(full_data['month'], errors='coerce').fillna(1).astype(int)
        full_data['date'] = pd.to_datetime(full_data[['year', 'month']].assign(day=1), errors='coerce')

        # Pivot from long to wide format
        wide_data = full_data.pivot_table(
            index=[self.polygon_id_col, 'year', 'month', 'date'],
            columns='land_use_name',
            values='land_use_percentage',
            fill_value=0
        ).reset_index()
        wide_data.columns.name = None

        # === Drop 'date' for the merge version ===
        merge_data = wide_data.drop(columns=['date'])

        expected_polygons = self.merged_data[self.polygon_id_col].unique()
        merge_data = merge_data[
            merge_data[self.polygon_id_col].isin(expected_polygons)
        ]

        self.merged_data = pd.merge(
            self.merged_data,
            merge_data,
            on=[self.polygon_id_col, 'year', 'month'],
            how='left'
        )

        print("Land use data merged successfully.")

        # Recreate the canonical date
        self.merged_data['date'] = pd.to_datetime(
            self.merged_data['year'].astype(int).astype(str) + '-' +
            self.merged_data['month'].astype(int).astype(str).str.zfill(2) + '-01',
            errors='coerce'
        )


        # === Plotting Section ===
        if not generate_plot:
            return

        land_use_classes = [col for col in wide_data.columns if col not in [
            'year', 'month', 'date', self.polygon_id_col]]

        if land_use_classes:
            avg_land_use = wide_data.groupby('date')[land_use_classes].mean().reset_index()
            avg_land_use.columns = ['date'] + [f"avg_{c}" for c in land_use_classes]

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
                            # conflict_100km_new="ACLED_conflict_12m_running_sum_by_polygon_100kmcutoff_2021_08_to_2025_02.pkl",
                            conflict_100km_new=None,                            
                            conflict_500km_old="ACLED_conflict_12m_running_sum_by_polygon_500kmcutoff_2005_2010.csv",
                            # conflict_500km_new="conflict_12m_running_sum_by_polygon_500kmcutoff_2024_2025.csv",
                            conflict_500km_new=None,                            
                            filter_from_year=2021,
                            generate_plot=True, save_plot=True, plot=True,
                            output_dir="covariates_graphs"):
        """
        Merges conflict data (100km and 500km), filters recent years, 
        and optionally plots cumulative conflict and fatalities.
        """
        print("Merging conflict data (100km and 500km radius)...")

        # --- 100km Conflict Data ---
        c100_old = pd.read_pickle(os.path.join(self.input_path, conflict_100km_old))
        # c100_path_new = os.path.join(self.input_path, conflict_100km_new)

        if conflict_100km_new is not None:
            c100_path_new = os.path.join(self.input_path, conflict_100km_new)

            if os.path.exists(c100_path_new):
                try:
                    c100_new = pd.read_pickle(c100_path_new)
                    c100 = pd.concat([c100_old, c100_new], ignore_index=True)
                    print("New 100km conflict data found and merged.")

                    if 'year' in c100.columns and 'month' in c100.columns:
                        c100['date'] = pd.to_datetime(
                            c100['year'].astype(str) + '-' + c100['month'].astype(str) + '-01')
                        start_date = c100['date'].min().strftime('%Y_%m')
                        end_date = c100['date'].max().strftime('%Y_%m')
                    else:
                        raise ValueError("Columns 'year' and 'month' must be present in the data.")

                    # Build new filename
                    new_filename = f"ACLED_conflict_12m_100kmcutoff_{start_date}_Onwards.pkl"

                    # Save
                    output_path = os.path.join(self.input_path, new_filename)
                    c100.to_pickle(output_path)
                    print(f"Saved merged dataset to '{output_path}'.")
                        # Delete old and new files
                    # try:
                    #     os.remove(os.path.join(self.input_path, conflict_100km_old))
                    #     print(f"Deleted old file: {os.path.join(self.input_path, conflict_100km_old)}")
                    # except Exception as e:
                    #     print(f"Could not delete old file: {e}")

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
        c100_filtered = c100[c100['year'] >= filter_from_year]

        # Drop redundant date column before merging
        if 'date' in c100_filtered.columns:
            c100_filtered = c100_filtered.drop(columns=['date'])

        expected_polygons = self.merged_data[self.polygon_id_col].unique()
        c100_filtered = c100_filtered[
            c100_filtered[self.polygon_id_col].isin(expected_polygons)
        ]

        self.merged_data = pd.merge(
            self.merged_data, c100_filtered,
            on=[self.polygon_id_col, 'year', 'month'],
            how='left'
        )

        # --- 500km Conflict Data ---
        c500_old = pd.read_pickle(os.path.join(self.input_path, conflict_500km_old))
        # c500_path_new = os.path.join(self.input_path, conflict_500km_new)

        if conflict_500km_new is not None:
            c500_path_new = os.path.join(self.input_path, conflict_500km_new)

            if os.path.exists(c500_path_new):
                try:
                    c500_new = pd.read_pickle(c500_path_new)
                    c500 = pd.concat([c500_old, c500_new], ignore_index=True)
                    print("New 500km conflict data found and merged.")


                    if 'year' in c500.columns and 'month' in c500.columns:
                        c500['date'] = pd.to_datetime(
                            c500['year'].astype(str) + '-' + c500['month'].astype(str) + '-01')
                        start_date = c500['date'].min().strftime('%Y_%m')
                        end_date = c500['date'].max().strftime('%Y_%m')
                    else:
                        raise ValueError("Columns 'year' and 'month' must be present in the data.")

                    # Build new filename
                    new_filename = f"ACLED_conflict_12m_500kmcutoff_{start_date}_Onwards.pkl"

                    # Save
                    output_path = os.path.join(self.input_path, new_filename)
                    c500.to_pickle(output_path)
                    print(f"Saved merged dataset to '{output_path}'.")
                        # Delete old and new files
                    # try:
                    #     os.remove(os.path.join(self.input_path, conflict_500km_old))
                    #     print(f"Deleted old file: {os.path.join(self.input_path, conflict_500km_old)}")
                    # except Exception as e:
                    #     print(f"Could not delete old file: {e}")

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
        c500_filtered = c500[c500['year'] >= filter_from_year]

        # Drop redundant date column before merging
        if 'date' in c500_filtered.columns:
            c500_filtered = c500_filtered.drop(columns=['date'])

        expected_polygons = self.merged_data[self.polygon_id_col].unique()
        c500_filtered = c500_filtered[
            c500_filtered[self.polygon_id_col].isin(expected_polygons)
        ]

        self.merged_data = pd.merge(
            self.merged_data, c500_filtered,
            on=[self.polygon_id_col, 'year', 'month'],
            how='left'
        )

        print("Conflict data merged successfully.")

        # Canonical date rebuild
        self.merged_data['date'] = pd.to_datetime(
        self.merged_data['year'].astype(int).astype(str) + '-' +
        self.merged_data['month'].astype(int).astype(str).str.zfill(2) + '-01',
        errors='coerce'
        )


        if not generate_plot:
            return

        # === Plotting Conflict Trends ===
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
        Extends the time series for each polygon by `months_forward` months after 
        the last non-NaN MUAC/wasting observation.
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

        # Only keep new rows not already present
        existing = self.merged_data[['Ward', 'date']]
        future_df = future_df.merge(existing, on=[self.polygon_id_col, 'date'], how='left', indicator=True)
        future_df = future_df[future_df['_merge'] == 'left_only'].drop(columns=['_merge'])

        # Recover missing metadata like County, SubCounty, etc.
        metadata_cols = ['County', 'SubCounty', 'LivelihoodZone']  # adjust if needed

        # Get unique metadata per Ward
        ward_metadata = (
            self.merged_data
            .dropna(subset=metadata_cols)
            .drop_duplicates(subset=[self.polygon_id_col])[ [self.polygon_id_col] + metadata_cols ]
        )

        # Merge metadata into future_df
        future_df = future_df.merge(ward_metadata, on=self.polygon_id_col, how='left')

        self.merged_data = pd.concat([self.merged_data, future_df], ignore_index=True)
        print(f"Added {len(future_df)} future rows.")

    def generate_lags_for_dynamic_variables(self, max_lag=12):
        """
        Generates up to `max_lag` months of lags for dynamic variables 
        in the merged dataset.
        """
        print(f"Generating lags up to {max_lag} months for dynamic variables...")

        # Ensure proper sorting
        self.merged_data = self.merged_data.sort_values(by=[self.polygon_id_col, 'year', 'month']).reset_index(drop=True)

        # Rename for consistency
        self.merged_data = self.merged_data.rename(columns={
            'avg_temp_month': 'avg_temp', 
        })

        # Define variable groups
        climate_vars = ['avg_temp', 'hot_days', 'cold_days', 'consec_hot_days', 'consec_cold_days', 
                        'precip_total', 'wet_days', 'dry_days', 'consec_wet_days', 'consec_dry_days']
        
        conflict_vars = ['conflict', 'fatalities']
        
        veg_index_vars = ['NDVI', 'EVI']
        
        land_use_classes = ['bare', 'built', 'crops', 'flooded_vegetation', 
                            'grass', 'shrub_and_scrub', 'trees', 'water']

        # Build dynamic variable list
        dynamic_col_list = []
        dynamic_col_list += [f"{var}_z_score" for var in climate_vars]
        dynamic_col_list += [f"{var}_{radius}" for var in conflict_vars for radius in ['100km', '500km']]
        dynamic_col_list += [f"{var}_12m_{radius}" for var in conflict_vars for radius in ['100km', '500km']]
        dynamic_col_list += [f"{var}_mean" for var in veg_index_vars]
        dynamic_col_list += [f"{var}_z_score" for var in veg_index_vars]
        dynamic_col_list += [f"{var}_z_score_{cls}" for var in veg_index_vars for cls in ['crops', 'grass', 'shrub_and_scrub', 'trees']]
        dynamic_col_list += land_use_classes
        #dynamic_col_list += [f"{var}_diff_{n}" for var in land_use_classes for n in ['3m', '6m', '9m', '12m', '24m']]

        print(f"Total dynamic variables for lagging: {len(dynamic_col_list)}")

        # Generate lags
        lagged_columns = {}
        for var in dynamic_col_list:
            if var in self.merged_data.columns:
                for lag in range(1, max_lag + 1):
                    lagged_columns[f'{var}_lag_{lag}'] = self.merged_data.groupby(self.polygon_id_col)[var].shift(lag)
            else:
                print(f"Warning: Variable '{var}' not found in dataset, skipping lag generation.")

        # Add lags to dataset
        self.merged_data = pd.concat([self.merged_data, pd.DataFrame(
            lagged_columns, index=self.merged_data.index)], axis=1)

        # === Add lagged deltas for land use classes ===
        for cls in land_use_classes:
            if cls in self.merged_data.columns:
                for delta in [1, 2, 3, 6]:
                    col = f"{cls}"
                    delta_col = f"{cls}_delta_lag_{delta}"
                    self.merged_data[delta_col] = (
                        self.merged_data.groupby(self.polygon_id_col)[col]
                        .diff(delta)
                        .shift(1)  # Lag the delta for forecasting safety
                    )
            else:
                print(f"Warning: Land use class '{cls}' not found in dataset, skipping delta computation.")

        self.merged_data = self.merged_data.copy()
        print(f"Lag and delta generation completed. Dataset now has {self.merged_data.shape[1]} columns.")

    def generate_seasonal_features_and_lags(self, season1, season2):
        """
        Generates seasonal statistics (max, total, avg) and 1-year 
        lags for climate, vegetation, and land use variables.
        """

        print("Generating seasonal features and lags...")

        # Define variable groups
        climate_vars = ['avg_temp', 'hot_days', 'cold_days', 'consec_hot_days', 'consec_cold_days',
                        'precip_total', 'wet_days', 'dry_days', 'consec_wet_days', 'consec_dry_days']
        veg_index_vars = ['NDVI', 'EVI']
        land_use_classes = ['bare', 'built', 'crops', 'flooded_vegetation',
                            'grass', 'shrub_and_scrub', 'trees', 'water']

        seasonal_variables = []
        seasonal_variables += [f"{var}_z_score" for var in climate_vars]
        seasonal_variables += [f"{var}_mean" for var in veg_index_vars]
        seasonal_variables += [f"{var}_z_score" for var in veg_index_vars]
        seasonal_variables += [f"{var}_z_score_{cls}" for var in veg_index_vars for cls in ['crops', 'grass', 'shrub_and_scrub', 'trees']]
        seasonal_variables += land_use_classes

        id_col = self.polygon_id_col
        df = self.merged_data.copy()


            # === Aggregate each season ===
        def process_season(df, months, season_name):
                df_season = df[df['month'].isin(months)].copy()
                df_season['season_year'] = df_season['year']
                seasonal_agg = df_season.groupby(
                    [id_col, 'season_year'])[seasonal_variables].agg(['max', 'sum', 'mean'])
                seasonal_agg.columns = [f"{var}_{season_name}_{stat}_prev" for var, stat in seasonal_agg.columns]
                return seasonal_agg.reset_index()

        season1_df = process_season(df, season1, 'season1')
        season2_df = process_season(df, season2, 'season2')

        # === Assign robust effective season years ===
        # For short rains (OND): always previous year
        df['effective_season2_year'] = df['year'] - 1

            # For long rains (MAM): previous year until May, current year from June onward
        df['effective_season1_year'] = df['year']
        df.loc[df['month'] < 6, 'effective_season1_year'] -= 1

        # === Merge seasonal aggregates ===
        df = df.merge(
                season1_df.rename(columns={'season_year': 'effective_season1_year'}),
                on=[id_col, 'effective_season1_year'],
                how='left'
            )

        df = df.merge(
                season2_df.rename(columns={'season_year': 'effective_season2_year'}),
                on=[id_col, 'effective_season2_year'],
                how='left'
            )

            # === Add 1-year lags ===
        for season in ['season1', 'season2']:
                for var in seasonal_variables:
                    for stat in ['max_prev', 'avg_prev']:
                        col = f"{var}_{season}_{stat}_prev"
                        lag_col = f"{var}_{season}_lag"
                        if col in df.columns:
                            df[lag_col] = df.groupby(id_col)[col].shift(1)

        self.merged_data = df
        print("✅ Seasonal feature generation complete.")


    def generate_wasting_lags(self, last_data_date='2025-03-01'):
        """
        Adds 12-month lags for wasting-related variables, plus lagged 1-, 2-, 3-, and 6-month
        delta (difference) features. Retains future dates with missing values beyond
        the last available collection period.

        Parameters:
            last_data_date (str): Final date (YYYY-MM-DD) with confirmed data collection.
        """
        print(f"Generating wasting lags and lagged deltas with missing values allowed after {last_data_date}...")

        # Convert to datetime
        last_data_date = pd.to_datetime(last_data_date)

        # Sort and reset
        self.merged_data = self.merged_data.sort_values(by=['Ward', 'year', 'month']).reset_index(drop=True)

        # Generate date column if needed
        if 'date' not in self.merged_data.columns:
            self.merged_data['date'] = pd.to_datetime(self.merged_data[['year', 'month']].assign(day=1))

        # Drop rows where wasting is missing AND date is before the last data collection date
        condition = ~((self.merged_data['wasting'].isna()) & (self.merged_data['date'] <= last_data_date))
        self.merged_data = self.merged_data[condition].reset_index(drop=True)

        # Create lags
        for lag in range(1, 13):
            self.merged_data[f'wasting_lag_{lag}'] = self.merged_data.groupby('Ward')['wasting'].shift(lag)
            self.merged_data[f'wasting_obs_{lag}'] = self.merged_data.groupby('Ward')['wasting_count'].shift(lag)
            self.merged_data[f'wasting_risk_lag_{lag}'] = self.merged_data.groupby('Ward')['wasting_risk'].shift(lag)

        # Create lagged deltas for forecasting (difference then shifted)
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

            Parameters:
                output_path (str): Directory to save the dataset.
                filename (str): File name to use. Defaults to a standard name.
            """
            full_path = os.path.join(output_path, filename)
            self.merged_data.to_pickle(full_path)
            print(f"Dataset saved to: {full_path}")

    def plot_seasonal_variable(self, var_name, season='season2', 
                                output_folder="covariates_graphs/seasonal_graphs",
                                data_span = '2021_01_to_2025_03'):
            """
            Plots the average value of a specified seasonal variable over time.
            
            Parameters:
                var_name (str): The base variable name to plot (e.g., 'EVI_z_score_grass').
                season (str): One of 'season2' or 'season1'. Appends `_avg_prev` internally.
                output_folder (str or None): If provided, saves the plot to this folder.
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
    polygon_id_col = polygon_id_col
)

# Run steps
builder.join_datasets_and_eliminate_duplicates()
builder.eliminate_prevalences_with_insufficient_obs(min_num_obs_ward=35)
builder.clean_for_data_continuity(total_months_min=6, max_gap_size=3)

last_year = int(new_muac_data["Year"].max())
analysis_years = (2021, last_year)

print(f"Detected analysis years: {analysis_years}")
# Merge static & dynamic variables

# filename = f"era5_stats_{start_Month_Year}_to_{end_Month_Year}.pkl"

builder.merge_travel_time()

builder.merge_population_density(
    population_file_name="population_density_2005_2020.csv", map_year=2020)

builder.merge_temperature_data(main_file="era5_temperature_stats_2004_01_Onwards.pkl",
                            # new_file=f"era5_stats_{start_Month_Year}_to_{end_Month_Year}.pkl",
                            new_file=None,                           
                            baseline_years=(2004, 2020),
                            analysis_years=analysis_years)

builder.merge_precipitation_data(main_file="chirps_precipitation_stats_2004_01_Onwards.pkl",
                                # new_file=f"chirps_stats_{start_Month_Year}_to_{end_Month_Year}.pkl",
                                new_file=None,                                
                                baseline_years=(2004, 2020),
                                analysis_years=analysis_years)

builder.merge_ndvi_evi_data(main_file="modis_ndvi_evi_stats_2016_01_Onwards.pkl",
                            # new_file=f"modis_ndvi_evi_stats_{start_Month_Year}_to_{end_Month_Year}.pkl",
                            new_file=None,
                            baseline_years=(2016, 2020),
                            analysis_years=analysis_years)

builder.merge_ndvi_evi_by_land_use(main_file="modis_evi_by_land_use_stats_2016_01_Onwards.pkl",
                            # new_file=f"modis_evi_by_land_use_stats_{start_Month_Year}_to_{end_Month_Year}.pkl",
                            new_file=None,                            
                            baseline_years=(2016, 2020),
                            analysis_years=analysis_years)

builder.merge_land_use_data(main_file="land_use_pct_stats_2016_01_Onwards.pkl",
                            # new_file=f"land_use_pct_stats_{start_Month_Year}_to_{end_Month_Year}.pkl")
                            new_file=None)

builder.merge_conflict_data(conflict_100km_old="ACLED_conflict_12m_100kmcutoff_2021_08_Onwards.pkl",
                            # conflict_100km_new=f"ACLED_conflict_12m_running_sum_by_polygon_100kmcutoff_{end_Month_Year1}.pkl",
                            conflict_100km_new=None,
                            conflict_500km_old="ACLED_conflict_12m_500kmcutoff_2021_08_Onwards.pkl",
                            # conflict_500km_new=f"ACLED_conflict_12m_running_sum_by_polygon_500kmcutoff_{end_Month_Year1}.pkl",
                            conflict_500km_new=None,
                            filter_from_year=2021)

# Generate features
builder.extend_future_muac_periods(months_forward=3)
builder.generate_seasonal_features_and_lags(season1=[3,4,5], 
                                            season2=[10,11,12])
builder.generate_lags_for_dynamic_variables(max_lag=12)
# builder.generate_wasting_lags(last_data_date='2025-03-01')

temp_muac = new_muac_data.rename(columns={'month_num': 'month', 'Year': 'year'})
temp_muac['date'] = pd.to_datetime(
    temp_muac[['year', 'month']].assign(day=1), 
    errors='coerce'
)
last_muac_date = temp_muac[temp_muac['wasting'].notna()]['date'].max().strftime("%Y-%m-%d")

builder.generate_wasting_lags(last_data_date=last_muac_date)

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

# Save with dynamic date label
builder.save_final_dataset(
    output_path=INPUT,
    filename=f'complete_ward_level_dataset_{date_label}_updated_on_{update_label}.pkl'
)

builder.save_final_dataset(
    output_path=INPUT,
    filename=f'complete_ward_level_dataset.pkl'
)

# Plot a specific seasonal variable
# season1 = long rain season (March, April, May)
# season2 = short rain season (October, November, December)
builder.plot_seasonal_variable(var_name="EVI_z_score_grass", season="season2")
