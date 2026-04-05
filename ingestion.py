@@ -0,0 +1,207 @@
import glob
import xarray as xr
import pandas as pd
import clickhouse_connect
import numpy as np
import concurrent.futures
import os
from tqdm import tqdm
import logging

# --- Configuration ---
DATA_DIRECTORY = 'Ocean/'
CLICKHOUSE_HOST = 'localhost'
CLICKHOUSE_USER = 'default'
CLICKHOUSE_PASSWORD = ''
BATCH_SIZE = 200
MAX_WORKERS = os.cpu_count() or 4

# --- Logging Configuration ---
LOG_FILE = 'data_ingestion.log'
PROCESSED_FILES_LOG = 'processed_files.log'


def setup_logging():
    """Configures logging to file and console."""
    # To avoid adding handlers multiple times in interactive environments
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )


def get_processed_files(log_path):
    """Reads the list of already processed files and normalizes their paths."""
    if not os.path.exists(log_path):
        return set()
    try:
        with open(log_path, 'r') as f:
            # NORMALIZED PATHS: Ensure all paths use forward slashes for consistent comparison
            return set(line.strip().replace('\\', '/') for line in f)
    except IOError as e:
        logging.error(f"Could not read processed files log {log_path}: {e}")
        return set()


def log_processed_batch(log_path, file_list):
    """Appends a list of successfully processed files to the log."""
    try:
        with open(log_path, 'a') as f:
            for file_path in file_list:
                f.write(f"{file_path}\n")
    except IOError as e:
        logging.error(f"Could not write to processed files log {log_path}: {e}")


def process_file(file_path):
    """
    Opens a single NetCDF file and extracts all profiles at once.
    This method is faster but less robust to malformed files.
    """
    try:
        with xr.open_dataset(file_path, engine="netcdf4") as ds:


            # --- Extract Profiles Data ---
            profile_vars = [
                'PLATFORM_NUMBER', 'CYCLE_NUMBER', 'DIRECTION', 'DATA_CENTRE', 'JULD',
                'JULD_LOCATION', 'LATITUDE', 'LONGITUDE', 'JULD_QC', 'POSITION_QC',
                'PROFILE_PRES_QC', 'PROFILE_TEMP_QC', 'PROFILE_PSAL_QC'
            ]
            # print(ds[profile_vars].dtypes)
            existing_profile_vars = [v for v in profile_vars if v in ds]
            profiles_df = ds[existing_profile_vars].to_dataframe(dim_order=['N_PROF']).reset_index(drop=True)
            profiles_df['MAX_LEVELS'] = ds.sizes.get('N_LEVELS', 0)

            # --- Extract Readings Data ---
            reading_vars = [
                'N_LEVELS', 'PLATFORM_NUMBER', 'CYCLE_NUMBER', 'PRES', 'TEMP', 'PSAL',
                'PRES_ADJUSTED', 'TEMP_ADJUSTED', 'PSAL_ADJUSTED',
                'PRES_ADJUSTED_QC', 'TEMP_ADJUSTED_QC', 'PSAL_ADJUSTED_QC',
                'PRES_ADJUSTED_ERROR', 'TEMP_ADJUSTED_ERROR', 'PSAL_ADJUSTED_ERROR'
            ]
            existing_reading_vars = [v for v in reading_vars if v in ds]
            readings_df = ds[existing_reading_vars].to_dataframe(dim_order=['N_PROF', 'N_LEVELS']).reset_index()
            if 'N_PROF' in readings_df.columns:
                readings_df = readings_df.drop(columns='N_PROF')

            # --- Extract Calibrations Data ---
            calibration_vars = [
                'PLATFORM_NUMBER', 'CYCLE_NUMBER', 'PARAMETER', 'SCIENTIFIC_CALIB_EQUATION',
                'SCIENTIFIC_CALIB_COEFFICIENT', 'SCIENTIFIC_CALIB_COMMENT', 'SCIENTIFIC_CALIB_DATE'
            ]
            existing_calib_vars = [v for v in calibration_vars if v in ds]
            calibrations_df = ds[existing_calib_vars].to_dataframe(
                dim_order=['N_PROF', 'N_CALIB', 'N_PARAM']).reset_index(drop=True)

            return profiles_df, readings_df, calibrations_df

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def main():
    setup_logging()

    # NORMALIZED PATHS: Ensure glob results use forward slashes for cross-platform compatibility
    all_files_raw = glob.glob(f'{DATA_DIRECTORY}/**/*.nc', recursive=True)
    all_files = sorted([p.replace('\\', '/') for p in all_files_raw])

    if not all_files:
        logging.info(f"No .nc files found in '{DATA_DIRECTORY}'. Exiting.")
        return

    processed_files = get_processed_files(PROCESSED_FILES_LOG)
    files_to_process = [f for f in all_files if f not in processed_files]

    logging.info(f"Found {len(all_files)} total files.")
    logging.info(f"Skipping {len(processed_files)} previously processed files.")
    logging.info(f"Processing {len(files_to_process)} new files.")

    if not files_to_process:
        logging.info("No new files to process. Exiting.")
        return

    file_chunks = [files_to_process[i:i + BATCH_SIZE] for i in range(0, len(files_to_process), BATCH_SIZE)]
    total_batches = len(file_chunks)
    logging.info(f"Data will be processed in {total_batches} batches of up to {BATCH_SIZE} files each.")

    try:
        client = clickhouse_connect.get_client(host=CLICKHOUSE_HOST, user=CLICKHOUSE_USER, password=CLICKHOUSE_PASSWORD)
    except Exception as e:
        logging.critical(f"Fatal: Could not connect to ClickHouse. Error: {e}")
        return

    for batch_num, batch_files in enumerate(file_chunks, 1):
        logging.info(f"--- Processing Batch {batch_num}/{total_batches} ---")
        batch_profiles_list, batch_readings_list, batch_calibrations_list = [], [], []

        # ThreadPoolExecutor is often better for I/O heavy tasks like reading many files
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results_iterator = executor.map(process_file, batch_files)
            desc = f"Batch {batch_num}/{total_batches}"
            for p_df, r_df, c_df in tqdm(results_iterator, total=len(batch_files), desc=desc):
                if not p_df.empty: batch_profiles_list.append(p_df)
                if not r_df.empty: batch_readings_list.append(r_df)
                if not c_df.empty: batch_calibrations_list.append(c_df)

        logging.info(f"Concatenating and cleaning data for batch {batch_num}...")

        try:
            # --- Process and Insert Profiles ---
            if batch_profiles_list:
                profiles = pd.concat(batch_profiles_list, ignore_index=True)
                profile_columns = profiles.select_dtypes(include='object').columns
                profile_mapping = {col: 'string' for col in profile_columns}
                profile_mapping['PLATFORM_NUMBER'] = 'uint32'
                profile_mapping['CYCLE_NUMBER'] = 'uint32'
                profiles = profiles.astype(profile_mapping)
                profiles.fillna('', inplace=True)
                logging.info(f"Inserting {len(profiles)} profiles...")
                client.insert_df('argo_profiles', profiles)

            # --- Process and Insert Readings ---
            if batch_readings_list:
                readings = pd.concat(batch_readings_list, ignore_index=True)
                readings = readings.dropna(subset=['PRES', 'TEMP', 'PSAL'], how='all')
                readings_mapping = readings.select_dtypes(include='object').columns
                readings[readings_mapping] = readings[readings_mapping].astype('string')
                mapping = {'PLATFORM_NUMBER': 'uint32', 'CYCLE_NUMBER': 'uint32'}
                readings = readings.astype({k: v for k, v in mapping.items() if k in readings.columns})
                readings_strings = readings.select_dtypes(include='string').columns
                readings[readings_strings] = readings[readings_strings].fillna('')
                logging.info(f"Inserting {len(readings)} readings...")
                client.insert_df('argo_readings', readings)

            # --- Process and Insert Calibrations ---
            if batch_calibrations_list:
                calibrations = pd.concat(batch_calibrations_list, ignore_index=True)
                calibrations.replace(r'^\s*$', np.nan, regex=True, inplace=True)
                calibrations = calibrations.dropna(subset=['PARAMETER'], how='any').fillna('')
                calibrations['SCIENTIFIC_CALIB_DATE'] = pd.to_datetime(calibrations['SCIENTIFIC_CALIB_DATE'],
                                                                       errors='coerce')
                mapping = {'PLATFORM_NUMBER': 'uint32', 'CYCLE_NUMBER': 'uint32'}
                calibrations = calibrations.astype({k: v for k, v in mapping.items() if k in calibrations.columns})
                logging.info(f"Inserting {len(calibrations)} calibrations...")
                client.insert_df('argo_calibrations', calibrations)

            # If all insertions for the batch succeed, log the files as processed.
            log_processed_batch(PROCESSED_FILES_LOG, batch_files)
            logging.info(f"--- Batch {batch_num} complete. {len(batch_files)} files logged as processed. ---")

        except Exception as db_error:
            logging.error(
                f"Failed to insert batch {batch_num} into ClickHouse. This batch will be retried on next run. Error: {db_error}")

    logging.info("All batches have been processed.")


if __name__ == "__main__":
    main()