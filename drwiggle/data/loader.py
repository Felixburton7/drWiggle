import pandas as pd
import os
import logging
import glob
from typing import Union, List, Dict, Optional, Tuple
import warnings

logger = logging.getLogger(__name__)

def find_data_file(data_dir: str, pattern: str) -> Optional[str]:
    """
    Finds a data file matching a pattern (potentially containing wildcards)
    within a specified directory.

    Args:
        data_dir: The absolute path to the directory to search in.
        pattern: The file pattern (e.g., "temperature_320_*.csv").

    Returns:
        The absolute path to the first matching file found, or None if no match.

    Raises:
        FileNotFoundError: If the data directory itself does not exist.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    search_pattern = os.path.join(data_dir, pattern)
    matching_files = glob.glob(search_pattern)

    if not matching_files:
        logger.warning(f"No files found matching pattern '{pattern}' in directory '{data_dir}'")
        return None

    if len(matching_files) > 1:
        logger.warning(f"Multiple files found for pattern '{pattern}'. Using the first one found: {matching_files[0]}")

    logger.info(f"Found data file: {matching_files[0]}")
    return matching_files[0]


def load_data(file_path_or_pattern: str, data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from a CSV file. Handles either a direct path or a pattern search within data_dir.

    Args:
        file_path_or_pattern: Absolute path to a CSV file, or a filename pattern.
        data_dir: Absolute path to the directory to search if `file_path_or_pattern` is a pattern.
                  Required if `file_path_or_pattern` is not an absolute path and not just a filename pattern.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the file/pattern cannot be resolved or the file doesn't exist.
        ValueError: If a pattern is given without a data_dir.
        Exception: For pandas CSV reading errors.
    """
    file_path: Optional[str] = None

    if os.path.isabs(file_path_or_pattern) and os.path.isfile(file_path_or_pattern):
        file_path = file_path_or_pattern
        logger.info(f"Loading data from absolute path: {file_path}")
    elif '*' in file_path_or_pattern or '?' in file_path_or_pattern: # Likely a pattern
        if not data_dir:
            raise ValueError("data_dir must be provided when using a file pattern.")
        file_path = find_data_file(data_dir, file_path_or_pattern)
        if file_path is None:
             raise FileNotFoundError(f"No file found matching pattern '{file_path_or_pattern}' in directory '{data_dir}'")
        logger.info(f"Loading data found via pattern: {file_path}")
    else: # Assume it's a relative path or just filename - needs data_dir
        if not data_dir:
             # Try relative to current directory as last resort
             potential_path = os.path.abspath(file_path_or_pattern)
             if os.path.isfile(potential_path):
                  file_path = potential_path
                  logger.warning(f"No data_dir provided. Assuming '{file_path_or_pattern}' is relative to CWD: {file_path}")
             else:
                raise ValueError("data_dir must be provided for relative paths or filename patterns.")
        else:
             potential_path = os.path.join(data_dir, file_path_or_pattern)
             if os.path.isfile(potential_path):
                 file_path = potential_path
                 logger.info(f"Loading data from file '{file_path_or_pattern}' in data_dir: {file_path}")
             else:
                 # Maybe it was a pattern after all? Try find_data_file
                 file_path = find_data_file(data_dir, file_path_or_pattern)
                 if file_path is None:
                     raise FileNotFoundError(f"File '{file_path_or_pattern}' not found in directory '{data_dir}' or as pattern.")
                 logger.info(f"Loading data found via pattern search for '{file_path_or_pattern}': {file_path}")


    if file_path is None or not os.path.exists(file_path):
        # This case should ideally be caught earlier, but acts as a safeguard
        raise FileNotFoundError(f"Could not resolve or find data file: {file_path_or_pattern}")

    try:
        # Basic CSV loading. Add options (sep, header, dtype) as needed.
        # Use low_memory=False for potentially mixed type columns if warnings occur
        with warnings.catch_warnings():
             # Can suppress DtypeWarning if necessary and understood
             # warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
             df = pd.read_csv(file_path, low_memory=False)
        logger.info(f"Successfully loaded data from {file_path}. Shape: {df.shape}, Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError: # Should be caught above, but re-raise clearly
         logger.error(f"Data file not found at the determined path: {file_path}")
         raise
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise

def load_rmsf_data(file_path: str, target_column: str) -> pd.Series:
    """
    Loads only the target RMSF column from a data file efficiently.

    Args:
        file_path: Absolute path to the data file (CSV).
        target_column: The name of the column containing RMSF values.

    Returns:
        pandas Series containing RMSF values.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the target column is not found in the file.
        Exception: For pandas reading errors.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info(f"Loading RMSF data (column: '{target_column}') from {file_path}...")
    try:
        # Optimize by reading only the necessary column for CSV
        if file_path.lower().endswith('.csv'):
            # Fast check if column exists by reading just the header
            header_df = pd.read_csv(file_path, nrows=0)
            if target_column not in header_df.columns:
                 # Attempt case-insensitive match as fallback
                 target_column_lower = target_column.lower()
                 matching_cols = [col for col in header_df.columns if col.lower() == target_column_lower]
                 if not matching_cols:
                      raise ValueError(f"Target column '{target_column}' not found in {file_path}. Available columns: {header_df.columns.tolist()}")
                 target_column = matching_cols[0] # Use the actual column name case
                 logger.warning(f"Using case-insensitive match for target column: '{target_column}'")

            # Read only the target column
            rmsf_series = pd.read_csv(file_path, usecols=[target_column]).squeeze("columns")
            if not isinstance(rmsf_series, pd.Series):
                 raise TypeError(f"Expected pd.Series after loading column '{target_column}', but got {type(rmsf_series)}.")
        else:
             # Fallback for non-CSV: load full DataFrame and select column
             logger.warning(f"File format not CSV ({file_path}). Loading full file to extract RMSF column.")
             df = load_data(file_path) # Use the general loader
             if target_column not in df.columns:
                  # Case-insensitive check
                 target_column_lower = target_column.lower()
                 matching_cols = [col for col in df.columns if col.lower() == target_column_lower]
                 if not matching_cols:
                      raise ValueError(f"Target column '{target_column}' not found in {file_path}. Available columns: {df.columns.tolist()}")
                 target_column = matching_cols[0]
                 logger.warning(f"Using case-insensitive match for target column: '{target_column}'")

             rmsf_series = df[target_column]

        # Validate data type
        if not pd.api.types.is_numeric_dtype(rmsf_series):
            logger.warning(f"RMSF column '{target_column}' is not numeric (dtype: {rmsf_series.dtype}). Attempting conversion.")
            try:
                rmsf_series = pd.to_numeric(rmsf_series, errors='coerce')
                nans_induced = rmsf_series.isnull().sum()
                if nans_induced > 0:
                     logger.warning(f"Conversion to numeric induced {nans_induced} NaN values in RMSF column.")
            except Exception as conv_err:
                logger.error(f"Failed to convert RMSF column '{target_column}' to numeric: {conv_err}")
                raise ValueError(f"RMSF column '{target_column}' could not be converted to numeric.")

        logger.info(f"Successfully loaded RMSF data. Count: {len(rmsf_series)}, NaN count: {rmsf_series.isnull().sum()}")
        return rmsf_series
    except ValueError as ve: # Re-raise specific errors
        logger.error(f"Value error loading RMSF data: {ve}")
        raise
    except FileNotFoundError:
        logger.error(f"RMSF data file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load RMSF data from {file_path}: {e}")
        raise
