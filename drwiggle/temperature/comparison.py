import logging
import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union

from drwiggle.config import get_temperature_config, get_binning_config, get_class_names
from drwiggle.utils.helpers import ensure_dir, load_object, save_object
from drwiggle.utils.metrics import generate_confusion_matrix_df
from drwiggle.utils.visualization import plot_metric_vs_temperature, plot_transition_matrix, _plotting_available

logger = logging.getLogger(__name__)

def find_result_files(
    base_output_dir: str,
    temperatures: List[Union[int, str]],
    model_name: Optional[str] = None,
    file_type: str = "predictions", # "predictions", "metrics", "binner"
    pattern_suffix: str = "_test_predictions.csv", # Example pattern
    temperature_prefix: str = "run_temp_" # Expected prefix for temp-specific dirs
) -> Dict[Union[int, str], str]:
    """
    Scans output directories for specific result files for given temperatures.

    Looks for files in temperature-specific subdirectories first (e.g., run_temp_320/),
    then falls back to the base output directory.

    Args:
        base_output_dir: The main output directory potentially containing subdirs per temp.
        temperatures: List of temperatures to look for.
        model_name: Specific model name if file is model-specific (used in filename pattern).
        file_type: Type of file to search for (used in logging).
        pattern_suffix: Suffix pattern of the file (e.g., "_metrics.csv", ".joblib").
        temperature_prefix: Prefix used for temperature-specific subdirectories.

    Returns:
        Dictionary mapping temperature to the found absolute file path.
    """
    result_files = {}
    logger.info(f"Scanning '{base_output_dir}' for {file_type} files (pattern suffix: '{pattern_suffix}') for temperatures: {temperatures}")

    if not os.path.isdir(base_output_dir):
        logger.error(f"Base output directory for scanning does not exist: {base_output_dir}")
        return result_files

    for temp in temperatures:
        # Construct expected temp-specific directory path
        temp_output_dir = os.path.join(base_output_dir, f"{temperature_prefix}{temp}")
        # Directories to search, prioritizing temp-specific
        search_dirs = [temp_output_dir, base_output_dir]

        found_path = None
        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                 # Only log if it's the temp-specific dir missing, base dir was checked earlier
                 if search_dir == temp_output_dir:
                     logger.debug(f"Temperature-specific directory not found: {search_dir}")
                 continue

            # Construct search pattern (handle cases where model_name is None)
            base_filename_pattern = f"{model_name or '*'}{pattern_suffix}"
            search_path = os.path.join(search_dir, base_filename_pattern)
            matching_files = glob.glob(search_path)

            if matching_files:
                 if len(matching_files) > 1:
                      logger.warning(f"Multiple {file_type} files found for temp {temp} in {search_dir} matching '{base_filename_pattern}'. Using first found: {matching_files[0]}")
                 # Return absolute path
                 found_path = os.path.abspath(matching_files[0])
                 logger.info(f"Found {file_type} file for temperature {temp} in {search_dir}: {found_path}")
                 break # Found in this directory, stop searching for this temp

        if found_path:
            result_files[temp] = found_path
        else:
            logger.warning(f"Could not find {file_type} file for temperature {temp} in expected locations: {search_dirs} with pattern '{base_filename_pattern}'")

    return result_files


def calculate_transition_matrix(
     predictions1: pd.Series, # Predictions at T1
     predictions2: pd.Series, # Predictions at T2 (must match index with T1)
     num_classes: int,
     class_names: Optional[List[str]] = None
 ) -> Optional[pd.DataFrame]:
     """
     Calculates a class transition matrix between two sets of predictions using generate_confusion_matrix_df.
     Requires predictions Series to be aligned (e.g., same residues/index).

     Args:
         predictions1: Series of predicted classes at Temperature 1.
         predictions2: Series of predicted classes at Temperature 2.
         num_classes: Total number of classes.
         class_names: Optional list of class names for labeling the matrix axes.

     Returns:
         DataFrame representing the transition matrix (counts), or None on error. Rows=T1, Cols=T2.
     """
     if predictions1.shape != predictions2.shape:
         logger.error("Prediction Series must have the same shape for transition matrix calculation.")
         return None
     if not predictions1.index.equals(predictions2.index):
         logger.warning("Prediction Series indices do not match. Results may be incorrect if rows don't represent the same items. Ensure alignment before calling.")
         # Attempting to proceed, but results might be meaningless

     # Use the confusion matrix utility, treating T1 as 'true' and T2 as 'pred'
     logger.debug(f"Calculating transition matrix for {len(predictions1)} aligned predictions.")
     # Create a dummy config just for passing class names if needed
     temp_config = {"evaluation": {"class_names": {i: name for i, name in enumerate(class_names)}} if class_names else {}}

     matrix_df = generate_confusion_matrix_df(predictions1, predictions2, config=temp_config) # Use helper

     if matrix_df is not None:
          # Rename axes for clarity
          matrix_df.index.name = "Class at T1"
          matrix_df.columns.name = "Class at T2"
          logger.debug("Transition matrix calculated successfully.")
     else:
          logger.error("Failed to calculate transition matrix using generate_confusion_matrix_df.")

     return matrix_df


def run_temperature_comparison_analysis(config: Dict[str, Any], model_name: Optional[str] = None):
    """
    Main function to perform temperature comparison analysis.

    Loads predictions and metrics from previous runs (identified by temperature)
    and calculates comparative statistics and generates plots.

    Args:
        config: The main configuration dictionary (already loaded and resolved).
        model_name: Specific model to compare. If None, attempts to find results for all enabled models.
                   Currently, analysis focuses on one model at a time or combined metrics.
    """
    temp_config = get_temperature_config(config)
    if not temp_config.get("comparison", {}).get("enabled", False):
        logger.info("Temperature comparison is disabled in the configuration ('temperature.comparison.enabled=false').")
        return

    base_output_dir = config['paths']['output_dir'] # Base directory where results are stored
    # Create a dedicated subdir for comparison results
    comparison_output_dir = os.path.join(base_output_dir, "temperature_comparison")
    ensure_dir(comparison_output_dir)

    temperatures = temp_config.get("available", [])
    if not temperatures or len(temperatures) < 2:
        logger.warning("Temperature comparison requires at least two temperatures defined in config ('temperature.available').")
        return

    # Sort temperatures numerically if possible for plotting order
    try:
        # Convert to float, handle strings like 'average' by placing them last
        sorted_temps = sorted(temperatures, key=lambda x: float(x) if str(x).replace('.', '', 1).isdigit() else float('inf'))
    except ValueError:
        logger.warning("Could not sort temperatures numerically. Using order from config.")
        sorted_temps = temperatures

    logger.info(f"--- Running Temperature Comparison ---")
    logger.info(f"Model focus: {model_name or 'All available models'}")
    logger.info(f"Temperatures: {sorted_temps}")
    logger.info(f"Results directory: {comparison_output_dir}")

    # --- Load necessary data ---
    # We need predictions for transition matrices and metrics for performance plots.
    # Handle case where model_name is not specified - try to load combined metrics.

    # 1. Load Metrics
    all_metrics_list = []
    # Try finding a combined summary file first (e.g., from multiple runs saved together)
    # Or look for model-specific metrics files per temperature run.
    # Let's assume a common pattern: evaluation_summary.csv or {model}_metrics.csv
    # If model_name is given, prioritize that pattern.
    metrics_suffix = f"{model_name}_evaluation_summary.csv" if model_name else "evaluation_summary.csv"
    metric_files = find_result_files(base_output_dir, sorted_temps, None, "metrics", metrics_suffix) # Search without model name in pattern?

    # If model-specific file not found, try a general one
    if not metric_files and model_name:
         metrics_suffix_generic = "evaluation_summary.csv"
         metric_files = find_result_files(base_output_dir, sorted_temps, None, "metrics", metrics_suffix_generic)

    for temp in sorted_temps:
        if temp in metric_files:
            path = metric_files[temp]
            try:
                metrics_df = pd.read_csv(path)
                # Check if 'model' column exists, if not, assume it's for the specified model_name
                if 'model' not in metrics_df.columns and model_name:
                    metrics_df['model'] = model_name
                elif 'model' not in metrics_df.columns and not model_name:
                     logger.warning(f"Metrics file {path} has no 'model' column and no model_name specified. Skipping.")
                     continue

                metrics_df['temperature'] = temp # Add temperature column for merging
                all_metrics_list.append(metrics_df)
                logger.info(f"Loaded metrics summary for temperature {temp} from {path}")
            except Exception as e:
                logger.warning(f"Failed to load metrics file {path} for temp {temp}: {e}")
        else:
             logger.warning(f"Metrics file not found for temperature {temp} using suffixes '{metrics_suffix}' or 'evaluation_summary.csv'.")


    combined_metrics_df = pd.concat(all_metrics_list, ignore_index=True) if all_metrics_list else pd.DataFrame()

    if not combined_metrics_df.empty:
         # Save the combined metrics dataframe
         combined_metrics_path = os.path.join(comparison_output_dir, f"{model_name or 'all'}_combined_metrics.csv")
         combined_metrics_df.to_csv(combined_metrics_path, index=False)
         logger.info(f"Combined metrics saved to {combined_metrics_path}")
    else:
         logger.warning("No metrics data loaded. Cannot generate metric comparison plots.")


    # 2. Load Predictions (only if transition matrices needed and model_name specified)
    plot_transitions = temp_config.get("comparison", {}).get("plot_transition_matrix", False) and _plotting_available
    all_predictions: Dict[Union[int, str], pd.DataFrame] = {}
    if plot_transitions:
        if not model_name:
            logger.warning("Cannot calculate transition matrices without a specific 'model_name' specified.")
            plot_transitions = False # Disable if no model specified
        else:
            pred_suffix = f"{model_name}_test_predictions.csv"
            prediction_files = find_result_files(base_output_dir, sorted_temps, None, "predictions", pred_suffix) # Search without model name in pattern first?
            if not prediction_files: # Try with model name explicitly if first failed
                 prediction_files = find_result_files(base_output_dir, sorted_temps, model_name, "predictions", "_test_predictions.csv")


            if len(prediction_files) < 2:
                 logger.warning(f"Found predictions for fewer than 2 temperatures for model '{model_name}'. Cannot calculate transitions.")
                 plot_transitions = False # Disable if not enough data
            else:
                # Load predictions and align them
                loaded_preds_list = []
                for temp, path in prediction_files.items():
                    try:
                        # Load necessary columns and create unique ID
                        preds_df = pd.read_csv(path, usecols=lambda col: col.lower() in ['domain_id', 'resid', 'predicted_class'])
                        preds_df['unique_id'] = preds_df['domain_id'].astype(str) + "_" + preds_df['resid'].astype(str)
                        preds_df = preds_df.set_index('unique_id')
                        # Rename prediction column to include temperature
                        preds_df = preds_df.rename(columns={'predicted_class': f'pred_{temp}'})
                        loaded_preds_list.append(preds_df[[f'pred_{temp}']]) # Select only the prediction column
                        logger.info(f"Loaded {len(preds_df)} predictions for model '{model_name}' at temperature {temp}.")
                    except Exception as e:
                        logger.warning(f"Failed to load or process predictions file {path} for temp {temp}: {e}")

                # Merge predictions based on the unique residue index
                if len(loaded_preds_list) >= 2:
                     aligned_preds_df = pd.concat(loaded_preds_list, axis=1, join='inner') # Use inner join to keep only common residues
                     logger.info(f"Aligned predictions across {len(prediction_files)} temperatures. Found {len(aligned_preds_df)} common residues.")
                     if aligned_preds_df.empty:
                          logger.warning("No common residues found across all loaded prediction files. Cannot calculate transitions.")
                          plot_transitions = False
                else:
                     aligned_preds_df = pd.DataFrame() # Ensure it exists but is empty
                     plot_transitions = False


    # --- Perform Comparisons & Plotting ---

    # 1. Plot Metrics vs Temperature
    if not combined_metrics_df.empty and _plotting_available:
        metrics_to_plot = temp_config.get("comparison", {}).get("metrics", [])
        logger.info(f"Plotting metrics {metrics_to_plot} vs temperature...")
        # Filter combined metrics for the specific model if one was provided
        plot_metrics_df = combined_metrics_df
        if model_name:
             plot_metrics_df = combined_metrics_df[combined_metrics_df['model'] == model_name].copy()
             if plot_metrics_df.empty:
                  logger.warning(f"No metrics found for specified model '{model_name}' in loaded data.")


        if not plot_metrics_df.empty:
             for metric in metrics_to_plot:
                 if metric in plot_metrics_df.columns:
                      plot_path = os.path.join(comparison_output_dir, f"{model_name or 'all'}_metric_{metric}_vs_temp.png")
                      plot_metric_vs_temperature(plot_metrics_df, metric, plot_path)
                 else:
                      logger.warning(f"Metric '{metric}' requested for plotting not found in loaded metrics data.")
        else:
             logger.warning("No metrics data available for plotting after filtering.")


    # 2. Calculate and Plot Transition Matrices
    if plot_transitions and not aligned_preds_df.empty:
        num_classes = get_binning_config(config).get('num_classes', 5)
        class_names = list(get_class_names(config).values()) if get_class_names(config) else None
        logger.info(f"Calculating and plotting transition matrices for model '{model_name}'...")

        # Compare adjacent temperatures in the sorted list
        for i in range(len(sorted_temps) - 1):
            temp1 = sorted_temps[i]
            temp2 = sorted_temps[i+1]
            col1 = f'pred_{temp1}'
            col2 = f'pred_{temp2}'

            if col1 not in aligned_preds_df.columns or col2 not in aligned_preds_df.columns:
                 logger.warning(f"Missing prediction columns for transition between {temp1}K and {temp2}K. Skipping.")
                 continue

            preds1_aligned = aligned_preds_df[col1]
            preds2_aligned = aligned_preds_df[col2]

            logger.info(f"Calculating transition matrix between {temp1}K and {temp2}K ({len(preds1_aligned)} common residues).")

            try:
                 transition_matrix = calculate_transition_matrix(preds1_aligned, preds2_aligned, num_classes, class_names)

                 if transition_matrix is not None:
                      # Save matrix to CSV
                      matrix_path = os.path.join(comparison_output_dir, f"{model_name}_transition_{temp1}_to_{temp2}.csv")
                      transition_matrix.to_csv(matrix_path)
                      logger.info(f"Transition matrix saved to {matrix_path}")

                      # Plot normalized matrix
                      plot_path_norm = os.path.join(comparison_output_dir, f"{model_name}_transition_{temp1}_to_{temp2}_norm.png")
                      plot_transition_matrix(transition_matrix, str(temp1), str(temp2), plot_path_norm, normalize=True)

                      # Plot count matrix
                      plot_path_counts = os.path.join(comparison_output_dir, f"{model_name}_transition_{temp1}_to_{temp2}_counts.png")
                      plot_transition_matrix(transition_matrix, str(temp1), str(temp2), plot_path_counts, normalize=False)

            except Exception as e:
                 logger.error(f"Failed to calculate or plot transition matrix between {temp1}K and {temp2}K: {e}", exc_info=True)

    elif plot_transitions and aligned_preds_df.empty:
         logger.warning("Skipping transition matrix plotting as no aligned predictions were loaded.")

    logger.info(f"--- Temperature Comparison Finished ---")
    logger.info(f"Results saved in: {comparison_output_dir}")
