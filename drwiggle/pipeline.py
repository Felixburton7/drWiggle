import logging
import os
import time
import traceback
from typing import Dict, Any, Optional, List, Union, Tuple
import sys
import glob      # <<<--- Ensure this import is present

import pandas as pd
import numpy as np
import joblib # For saving/loading binners/models (used by helpers)

# Local Imports
from drwiggle.config import (
    load_config, get_model_config, get_enabled_features, get_class_names,
    is_pdb_enabled, get_binning_config, get_pdb_config, get_visualization_colors,
    get_split_config, get_system_config, get_pdb_feature_config
)
from drwiggle.data.loader import load_data, find_data_file, load_rmsf_data
from drwiggle.data.processor import process_features, split_data, prepare_data_for_model
from drwiggle.data.binning import get_binner, BaseBinner
# <<<--- Ensure get_model_class is imported --- >>>
from drwiggle.models import get_model_instance, get_enabled_models, BaseClassifier, get_model_class
from drwiggle.utils.metrics import evaluate_classification, generate_confusion_matrix_df, generate_classification_report_dict
from drwiggle.utils.visualization import (
    plot_bin_distribution, plot_confusion_matrix, plot_feature_importance,
    plot_class_distribution, plot_metric_vs_temperature, _plotting_available
    # plot_protein_visualization_data # Placeholder for PDB vis trigger
)
from drwiggle.utils.pdb_tools import parse_pdb, extract_pdb_features, generate_pymol_script, color_pdb_by_flexibility
from drwiggle.temperature.comparison import run_temperature_comparison_analysis # Avoid name clash
from drwiggle.utils.helpers import ensure_dir, save_object, load_object, timer # Use helpers

logger = logging.getLogger(__name__)

class Pipeline:
    """
    Main pipeline orchestrator for the drWiggle workflow.

    Handles data loading, processing, binning, model training, evaluation,
    prediction, and visualization based on the provided configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with a validated and resolved configuration.

        Args:
            config: The configuration dictionary, assumed to be loaded, merged,
                    templated, and path-resolved via `drwiggle.config.load_config`.
        """
        self.config = config
        self.binner: Optional[BaseBinner] = None
        # Store trained models: {model_name: model_instance}
        self.models: Dict[str, BaseClassifier] = {}
        # Store feature names used during training for consistency
        self.feature_names_in_: Optional[List[str]] = None
        self._prepare_directories()
        logger.info("drWiggle Pipeline initialized.")
        # Avoid logging full config at INFO level if it contains sensitive info or is too large
        # logger.debug(f"Pipeline using resolved config: {self.config}")


    def _prepare_directories(self):
        """Create output directories defined in the config if they don't exist."""
        logger.debug("Ensuring required directories exist...")
        paths_config = self.config.get('paths', {})
        required_dirs = ['output_dir', 'models_dir'] # Minimum required dirs
        if is_pdb_enabled(self.config): required_dirs.append('pdb_cache_dir')

        for key in required_dirs:
            path = paths_config.get(key)
            if path:
                try:
                    ensure_dir(path) # Use helper to create dir
                except Exception as e:
                    logger.error(f"Failed to create or access directory '{key}': {path}. Error: {e}")
                    # Decide whether to raise or just warn
                    raise # Critical failure if output dirs cannot be created
            else:
                logger.error(f"Required directory path '{key}' not found in configuration paths: {paths_config}")
                raise ValueError(f"Missing required directory path in config: paths.{key}")


    @timer # Time this method
    def load_and_process_data(self, data_path_or_pattern: Optional[str] = None) -> pd.DataFrame:
        """Loads data using load_data and processes features using process_features."""
        logger.info("--- Loading and Processing Data ---")
        data_dir = self.config.get('paths', {}).get('data_dir') # May be None if not specified
        file_pattern = self.config.get('dataset', {}).get('file_pattern') # Templated already

        # Determine input path: use specific path, pattern from config, or fail
        input_source = data_path_or_pattern
        if input_source is None:
            if not file_pattern:
                raise ValueError("No input data path/pattern provided via CLI (--input) and 'dataset.file_pattern' not set in config.")
            if not data_dir:
                 raise ValueError("Config 'paths.data_dir' must be set when using 'dataset.file_pattern' without explicit --input.")
            input_source = file_pattern # Use pattern from config
            logger.info(f"No explicit input path provided, using pattern from config: '{input_source}' in dir '{data_dir}'")
        else:
             logger.info(f"Using provided input path/pattern: '{input_source}'")
             # Check if data_dir is needed for relative path or pattern
             if not os.path.isabs(input_source) and ('*' in input_source or '?' in input_source or not os.path.dirname(input_source)):
                  if not data_dir: logger.warning(f"Input '{input_source}' looks like a pattern or relative filename, but 'paths.data_dir' is not set. Searching relative to CWD.")


        # Load data using the loader function
        try:
            df_raw = load_data(input_source, data_dir=data_dir)
        except FileNotFoundError as e:
            logger.error(f"Input data not found using source '{input_source}' and data_dir '{data_dir}': {e}")
            raise
        except ValueError as e:
             logger.error(f"Error determining input data source: {e}")
             raise

        # Process features using the processor module
        df_processed = process_features(df_raw, self.config)

        # Store feature names used for training (if this is the main data loading step)
        # This assumes process_features defines the final set of features.
        # We need X, y separated later, but store potential feature names now.
        # self.feature_names_in_ = df_processed.columns.tolist() # Store all columns for now? No, get from prepare_data

        logger.info(f"Data loading and processing finished. Shape: {df_processed.shape}")
        return df_processed


    @timer
    def calculate_and_apply_binning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates RMSF bin boundaries and adds a 'flexibility_class' column.

        Args:
            df: DataFrame containing the target RMSF column (defined in config).

        Returns:
            DataFrame with the added 'flexibility_class' column.
        """
        logger.info("--- Calculating and Applying RMSF Binning ---")
        target_col = self.config['dataset']['target'] # Already templated
        target_col_actual = None
        # Try to find the target column case-insensitively
        target_col_lower = target_col.lower()
        for col in df.columns:
            if col.lower() == target_col_lower:
                target_col_actual = col
                break
        if not target_col_actual:
            raise ValueError(f"Target RMSF column matching '{target_col}' not found in DataFrame columns: {df.columns.tolist()}")
        if target_col_actual != target_col:
             logger.warning(f"Using case-insensitive match for target column: '{target_col_actual}' (requested '{target_col}')")

        if df[target_col_actual].isnull().any():
             nan_count = df[target_col_actual].isnull().sum()
             logger.warning(f"Target RMSF column '{target_col_actual}' contains {nan_count} NaN values. These rows will likely be dropped or cause errors during binning/training.")
             # Optionally drop NaNs here? Or let binning handle it? Let binning handle for now.
             # df = df.dropna(subset=[target_col_actual])

        rmsf_values = df[target_col_actual].values

        # Get binner instance based on config
        self.binner = get_binner(self.config) # Factory function gets appropriate binner
        logger.info(f"Using binner type: {self.binner.__class__.__name__}")

        # Fit binner and transform data
        start_time_bin = time.time()
        try:
             class_labels = self.binner.fit_transform(rmsf_values)
        except Exception as e:
             logger.error(f"Error during binning process: {e}", exc_info=True)
             raise # Re-raise error after logging

        df_binned = df.copy()
        df_binned['flexibility_class'] = class_labels

        logger.info(f"Binning complete in {time.time() - start_time_bin:.2f} seconds.")
        boundaries = self.binner.get_boundaries()
        logger.info(f"Calculated bin boundaries: {[f'{b:.3f}' for b in boundaries] if boundaries else 'N/A'}")

        # Plot distribution if enabled and plotting available
        if _plotting_available and self.config.get('visualization', {}).get('plots', {}).get('bin_distribution', False):
             dist_plot_path = os.path.join(self.config['paths']['output_dir'], 'rmsf_distribution_with_bins.png')
             try:
                 plot_bin_distribution(rmsf_values, boundaries, self.config, dist_plot_path)
             except Exception as e:
                 logger.warning(f"Failed to plot RMSF distribution with bins: {e}", exc_info=True)

        # Save binner if configured
        if self.config['binning'].get('store_boundaries', True):
            binner_path = os.path.join(self.config['paths']['models_dir'], 'binner.joblib')
            try:
                self.binner.save(binner_path)
            except Exception as e:
                 logger.error(f"Failed to save binner state to {binner_path}: {e}", exc_info=True)
                 # Decide whether to raise error or just warn
                 # raise IOError(f"Failed to save binner state") from e

        return df_binned

    @timer
    def train(self, model_names: Optional[List[str]] = None, data_path: Optional[str] = None):
        """
        Full training pipeline: Load data -> Process -> Bin -> Split -> Train Models -> Save Models.

        Args:
            model_names: Optional list of specific model names (from config) to train.
                         If None, trains all enabled models.
            data_path: Optional path/pattern to specific input data file. Overrides config pattern.
        """
        logger.info("====== Starting Training Pipeline ======")
        # 1. Load and Process Data
        try:
            df_processed = self.load_and_process_data(data_path)
        except (FileNotFoundError, ValueError) as e:
             logger.error(f"Failed to load or process data: {e}")
             sys.exit(1) # Exit if data loading fails

        # 2. Calculate and Apply Binning (Fit binner on full processed dataset)
        try:
            df_binned = self.calculate_and_apply_binning(df_processed)
        except (ValueError, RuntimeError) as e:
             logger.error(f"Failed during RMSF binning: {e}")
             sys.exit(1)

        # 3. Split Data (Train/Validation/Test)
        try:
            train_df, val_df, test_df = split_data(df_binned, self.config)
        except ValueError as e:
             logger.error(f"Failed to split data: {e}")
             sys.exit(1)

        # 4. Prepare Data for Models (extract features and target)
        # Do this *after* splitting to avoid data leakage if scaling depends on data
        try:
            # Use the target 'flexibility_class' created by binning
            X_train, y_train, self.feature_names_in_ = prepare_data_for_model(train_df, self.config, target_col='flexibility_class')
            X_val, y_val, _ = prepare_data_for_model(val_df, self.config, target_col='flexibility_class', features=self.feature_names_in_)
            X_test, y_test, _ = prepare_data_for_model(test_df, self.config, target_col='flexibility_class', features=self.feature_names_in_)
        except (ValueError, KeyError) as e:
             logger.error(f"Failed to prepare data arrays for models: {e}", exc_info=True)
             sys.exit(1)

        logger.info(f"Data shapes: Train X={X_train.shape}, y={y_train.shape} | Val X={X_val.shape}, y={y_val.shape} | Test X={X_test.shape}, y={y_test.shape}")
        logger.info(f"Using {len(self.feature_names_in_)} features: {', '.join(self.feature_names_in_[:15])}{'...' if len(self.feature_names_in_)>15 else ''}")

        # 5. Determine Models to Train
        if not model_names:
            enabled_model_info = get_enabled_models(self.config)
            model_names = list(enabled_model_info.keys())
            if not model_names:
                 logger.error("No models are enabled in the configuration. Nothing to train.")
                 return # Or sys.exit(1) ?

        logger.info(f"Models selected for training: {model_names}")

        # 6. Train Each Selected Model
        training_successful = {} # Track success per model
        for model_name in model_names:
            logger.info(f"--- Training Model: {model_name} ---")
            start_time_model = time.time()
            try:
                # Get model instance (handles config extraction)
                model = get_model_instance(self.config, model_name)

                # Fit the model (HPO happens inside model.fit if enabled)
                model.fit(X_train, y_train, X_val, y_val) # Pass DataFrames/Series

                if model._fitted:
                     self.models[model_name] = model # Store fitted model instance
                     training_successful[model_name] = True
                     logger.info(f"Training {model_name} complete in {time.time() - start_time_model:.2f} seconds.")

                     # Save model immediately after successful training
                     model_path = os.path.join(self.config['paths']['models_dir'], f'{model_name}.joblib')
                     model.save(model_path) # Model's save method handles logging

                     # Plot feature importance if available and configured
                     if _plotting_available and self.config.get('visualization', {}).get('plots', {}).get('feature_importance', False):
                         try:
                             importances = model.get_feature_importance()
                             if importances:
                                  fi_plot_path = os.path.join(self.config['paths']['output_dir'], f'{model_name}_feature_importance.png')
                                  plot_feature_importance(importances, fi_plot_path, model_name=model_name) # Pass dict directly
                         except Exception as e:
                             logger.warning(f"Failed to plot feature importance for {model_name}: {e}", exc_info=True)
                else:
                     logger.error(f"Model {model_name} fitting reported as unsuccessful (model._fitted is False).")
                     training_successful[model_name] = False

            except NotImplementedError as e: # Catch HPO not implemented etc.
                logger.error(f"Configuration error for model {model_name}: {e}")
                training_successful[model_name] = False
            except Exception as e:
                logger.error(f"Critical error during training of model {model_name}: {e}", exc_info=True)
                logger.error(f"Traceback: {traceback.format_exc()}")
                training_successful[model_name] = False
                # Optionally continue to next model or exit? Continuing for now.

        logger.info("--- Training Phase Summary ---")
        for name, success in training_successful.items():
             logger.info(f"Model '{name}': {'Successfully Trained' if success else 'Training FAILED'}")
        logger.info("====== Training Pipeline Finished ======")


    @timer
    def evaluate(self, model_names: Optional[List[str]] = None, data_path: Optional[str] = None):
        """
        Evaluate trained models on the test set.

        Loads data (if needed), loads models, performs prediction, calculates metrics,
        and saves results/plots.

        Args:
            model_names: Optional list of model names to evaluate. If None, tries to find
                         and evaluate all models saved in the models directory.
            data_path: Optional path/pattern to specific data file for evaluation.
                       If None, uses the test split derived from the main data source.
        """
        logger.info("====== Starting Evaluation Pipeline ======")

        # 1. Determine Models to Evaluate
        models_dir = self.config['paths']['models_dir']
        if not model_names:
            # Find saved models in the directory (looking for .joblib as primary save format)
            found_files = glob.glob(os.path.join(models_dir, "*.joblib")) # Use glob here
            # Exclude binner file
            model_files = [f for f in found_files if not os.path.basename(f).startswith('binner')]
            # Exclude NN metadata file if naming convention is used
            model_files = [f for f in model_files if not os.path.basename(f).endswith('_meta.joblib')]
            if not model_files:
                 logger.error(f"No models specified and no primary '.joblib' model files found in {models_dir}. Cannot evaluate.")
                 return
            # Extract model names from filenames
            model_names = [os.path.splitext(os.path.basename(f))[0] for f in model_files]
            logger.info(f"No models specified via CLI. Found saved models to evaluate: {model_names}")
        else:
             logger.info(f"Evaluating specified models: {model_names}")

        # 2. Load Binner (required for consistent class labels if evaluating on external data)
        if not self.binner:
            binner_path = os.path.join(models_dir, 'binner.joblib')
            if os.path.exists(binner_path):
                try:
                    self.binner = BaseBinner.load(binner_path, config=self.config)
                    logger.info(f"Loaded binner ({self.binner.__class__.__name__}) from {binner_path}")
                except Exception as e:
                    logger.error(f"Failed to load required binner from {binner_path}: {e}", exc_info=True)
                    sys.exit(1)
            else:
                logger.error(f"Binner file not found at {binner_path}. Binner is required for evaluation. Run training first or ensure binner exists.")
                sys.exit(1)

        # 3. Load and Prepare Evaluation Data
        # If data_path is given, load and process that. Otherwise, need to regenerate test split.
        if data_path:
            logger.info(f"Evaluating using explicitly provided data: {data_path}")
            df_processed = self.load_and_process_data(data_path)
            # Apply binning using the loaded binner
            logger.info("Applying loaded binner to evaluation data...")
            rmsf_col = self.config['dataset']['target'] # Assumes target col name is consistent
            rmsf_col_actual = next((col for col in df_processed.columns if col.lower() == rmsf_col.lower()), None)
            if not rmsf_col_actual: raise ValueError(f"Target RMSF column '{rmsf_col}' not found in evaluation data.")
            df_processed['flexibility_class'] = self.binner.transform(df_processed[rmsf_col_actual].values)
            eval_df = df_processed # Use the entire processed & binned external dataset
        else:
            logger.info("Evaluating on the test split derived from the training data source.")
            # Need to reload, reprocess, re-bin, and re-split the original data
            # This ensures consistency but might be slow if data is large
            df_processed = self.load_and_process_data() # Load default data source
            df_binned = self.calculate_and_apply_binning(df_processed) # Re-bin (using loaded binner if available)
            _, _, eval_df = split_data(df_binned, self.config) # Get the test split
            logger.info(f"Using test split for evaluation ({len(eval_df)} rows).")

        if eval_df.empty:
             logger.error("Evaluation dataset is empty. Cannot proceed.")
             return

        # 4. Evaluate Each Model
        all_metrics = {}
        evaluation_results = [] # Store detailed results per model

        for model_name in model_names:
            logger.info(f"--- Evaluating Model: {model_name} ---")
            model = self.models.get(model_name) # Check if already in memory

            # Load model if not already loaded
            if not model:
                # Assume primary file is .joblib (holds RF object or NN metadata)
                model_path = os.path.join(models_dir, f'{model_name}.joblib')
                if os.path.exists(model_path):
                    try:
                        # <<<--- START CORRECTED MODEL LOADING --- >>>
                        # 1. Get the correct model class based on name
                        ModelClass = get_model_class(self.config, model_name) # Get e.g., RandomForestClassifier class type
                        if ModelClass is None:
                            # Log error and skip this model
                            logger.error(f"Could not determine model class for '{model_name}'. Skipping.")
                            continue # Go to the next model in the loop

                        # 2. Call the load method specific to that class
                        # This path (.joblib) works for RF and is the meta_path for NN
                        model = ModelClass.load(model_path, config=self.config) # Calls the correct .load()
                        # <<<--- END CORRECTED MODEL LOADING --- >>>

                        self.models[model_name] = model # Store loaded model
                        # Store feature names from loaded model if pipeline doesn't have them yet
                        # Access feature_names_in_ AFTER model is loaded
                        if model and self.feature_names_in_ is None and model.feature_names_in_:
                             self.feature_names_in_ = model.feature_names_in_
                             logger.info(f"Loaded feature names ({len(self.feature_names_in_)}) from model {model_name}.")
                    except Exception as e:
                        logger.error(f"Failed to load model {model_name} from {model_path}: {e}", exc_info=True)
                        continue # Skip evaluation for this model
                else:
                    logger.warning(f"Model {model_name} not found in memory or at {model_path}. Skipping evaluation.")
                    continue

            # Check if feature names are available (needed for data prep)
            if self.feature_names_in_ is None:
                 logger.error(f"Feature names required for evaluation of {model_name} are not available (neither from training run nor loaded model). Skipping.")
                 continue

            # Prepare evaluation data using the stored feature names
            try:
                X_eval, y_eval, _ = prepare_data_for_model(eval_df, self.config, target_col='flexibility_class', features=self.feature_names_in_)
            except (ValueError, KeyError) as e:
                 logger.error(f"Failed to prepare evaluation data for model {model_name}: {e}", exc_info=True)
                 continue

            if y_eval is None:
                logger.error(f"Target column 'flexibility_class' missing in evaluation data. Cannot evaluate {model_name}.")
                continue

            # Perform prediction
            try:
                y_pred = model.predict(X_eval)
                y_prob = None
                if hasattr(model, 'predict_proba'):
                    try:
                        y_prob = model.predict_proba(X_eval)
                    except NotImplementedError:
                         logger.debug(f"Model {model_name} does not support predict_proba.")
                    except Exception as e:
                         logger.warning(f"Error calling predict_proba for {model_name}: {e}")

            except Exception as e:
                logger.error(f"Error during prediction with model {model_name}: {e}", exc_info=True)
                continue # Skip metrics calculation if prediction fails

            # Calculate metrics
            metrics = evaluate_classification(y_eval, y_pred, self.config, y_prob, model_name)
            all_metrics[model_name] = metrics

            # Generate and save detailed reports/plots per model
            output_dir = self.config['paths']['output_dir']
            model_output_prefix = os.path.join(output_dir, f"{model_name}_eval")

            # Confusion Matrix
            if self.config.get("evaluation", {}).get("metrics", {}).get("confusion_matrix"):
                 cm_df = generate_confusion_matrix_df(y_eval, y_pred, config=self.config)
                 if cm_df is not None:
                     cm_path = f"{model_output_prefix}_confusion_matrix.csv"
                     cm_df.to_csv(cm_path)
                     logger.info(f"Confusion matrix saved to {cm_path}")
                     if _plotting_available and self.config.get('visualization', {}).get('plots', {}).get('confusion_matrix'):
                          cm_plot_path = f"{model_output_prefix}_confusion_matrix.png"
                          plot_confusion_matrix(cm_df, cm_plot_path, normalize=True) # Plot normalized version

            # Classification Report
            if self.config.get("evaluation", {}).get("metrics", {}).get("classification_report"):
                 report_dict = generate_classification_report_dict(y_eval, y_pred, config=self.config)
                 if report_dict:
                      report_path = f"{model_output_prefix}_classification_report.json"
                      save_object(report_dict, report_path) # Save dict using helper

            # Save test predictions with actual labels
            pred_df = pd.DataFrame({
                 'actual_class': y_eval.values,
                 'predicted_class': y_pred
             })
            # Add identifiers back from eval_df if possible
            id_cols = [col for col in ['domain_id', 'chain_id', 'resid', 'icode'] if col in eval_df.columns]
            if id_cols:
                 pred_df = pd.concat([eval_df[id_cols].reset_index(drop=True), pred_df], axis=1)

            if y_prob is not None:
                 prob_cols = [f'prob_class_{i}' for i in range(y_prob.shape[1])]
                 prob_df = pd.DataFrame(y_prob, columns=prob_cols)
                 pred_df = pd.concat([pred_df, prob_df], axis=1)

            pred_path = os.path.join(output_dir, f'{model_name}_test_predictions.csv') # Save in main output dir
            pred_df.to_csv(pred_path, index=False, float_format='%.4f')
            logger.info(f"Test predictions saved to {pred_path}")

            # Append results for summary table
            metrics['model'] = model_name
            evaluation_results.append(metrics)


        # 7. Save Overall Metrics Summary
        if evaluation_results:
            summary_df = pd.DataFrame(evaluation_results)
            # Reorder columns to put 'model' first
            cols = ['model'] + [col for col in summary_df.columns if col != 'model']
            summary_df = summary_df[cols]
            summary_path = os.path.join(self.config['paths']['output_dir'], 'evaluation_summary.csv')
            summary_df.to_csv(summary_path, index=False, float_format='%.4f')
            logger.info(f"Evaluation summary saved to {summary_path}")
        else:
             logger.warning("No evaluation results generated to save in summary.")


        logger.info("====== Evaluation Pipeline Finished ======")
        return all_metrics


    @timer
    def predict(self, data: Union[str, pd.DataFrame], model_name: Optional[str] = None, output_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Generate flexibility class predictions for new data using a specified or default model.

        Args:
            data: Input data as a file path/pattern or a pandas DataFrame.
                  Must contain the features required by the model.
            model_name: Name of the trained model to use (must exist in models_dir).
                        If None, attempts to load a default model (e.g., 'random_forest').
            output_path: Optional path to save the predictions CSV file. If None, returns DataFrame.

        Returns:
            DataFrame containing predictions (and probabilities if supported/enabled),
            or None if prediction fails or output_path is specified.
        """
        logger.info("====== Starting Prediction Pipeline ======")

        # 1. Load and Prepare Data
        if isinstance(data, (str, os.PathLike)):
             logger.info(f"Loading data for prediction from: {data}")
             # Assume data is already processed or contains necessary raw features
             df_input = self.load_and_process_data(data_path_or_pattern=str(data))
        elif isinstance(data, pd.DataFrame):
             logger.info("Using provided DataFrame for prediction.")
             # Assume DataFrame is already processed, or process it here?
             # For now, assume it contains necessary features directly.
             df_input = data.copy() # Work on a copy
        else:
            raise TypeError(f"Unsupported input data type for prediction: {type(data)}. Provide path or DataFrame.")

        if df_input.empty:
             logger.error("Input data for prediction is empty.")
             return None

        # 2. Load Model
        models_dir = self.config['paths']['models_dir']
        if not model_name:
            # Try loading a default model, e.g., RandomForest, if not specified
            default_model_to_try = "random_forest" # Make this configurable?
            model_name = default_model_to_try
            logger.warning(f"No model specified for prediction. Attempting to load default: '{model_name}'.")

        # Assume primary file is .joblib (holds RF object or NN metadata)
        model_path = os.path.join(models_dir, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            logger.error(f"Model file for '{model_name}' not found at {model_path}. Cannot predict.")
            return None

        try:
            # Check if model already loaded in pipeline instance
            model = self.models.get(model_name)
            if not model:
                 # <<<--- START CORRECTED MODEL LOADING (Predict) --- >>>
                 # 1. Get the correct model class based on name
                 ModelClass = get_model_class(self.config, model_name) # Get e.g., RandomForestClassifier class type
                 if ModelClass is None:
                      # Log error and exit
                      logger.error(f"Could not determine model class for '{model_name}'. Cannot predict.")
                      return None # Exit prediction

                 # 2. Call the load method specific to that class
                 model = ModelClass.load(model_path, config=self.config) # Calls the correct .load()
                 # <<<--- END CORRECTED MODEL LOADING (Predict) --- >>>
                 self.models[model_name] = model # Store loaded model
            # Ensure feature names are available from the loaded model
            if not model.feature_names_in_:
                 raise ValueError(f"Loaded model '{model_name}' does not contain feature names. Cannot ensure prediction consistency.")
            self.feature_names_in_ = model.feature_names_in_ # Use features from loaded model
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}' from {model_path}: {e}", exc_info=True)
            return None

        # 3. Prepare Dataframe for Prediction (Select features)
        try:
            # Ensure only necessary features in correct order are passed
            missing_features = set(self.feature_names_in_) - set(df_input.columns)
            if missing_features:
                 # Try to recalculate missing features if possible (e.g., process the input df)
                 logger.warning(f"Input data missing required features: {missing_features}. Attempting to process input DataFrame.")
                 df_input_processed = process_features(df_input, self.config)
                 # Check again after processing
                 missing_features = set(self.feature_names_in_) - set(df_input_processed.columns)
                 if missing_features:
                      raise ValueError(f"Input data still missing required features for model '{model_name}' after processing: {missing_features}")
                 df_to_use = df_input_processed
            else:
                 df_to_use = df_input # Use original if features already present

            X_pred, _, _ = prepare_data_for_model(df_to_use, self.config, target_col=None, features=self.feature_names_in_)

        except (ValueError, KeyError) as e:
            logger.error(f"Failed to prepare input data for prediction with model '{model_name}': {e}", exc_info=True)
            return None

        # 4. Perform Prediction
        logger.info(f"Predicting using model '{model_name}' on {len(X_pred)} samples...")
        try:
            y_pred = model.predict(X_pred)
            y_prob = None
            if self.config.get("cli_options", {}).get("predict_probabilities", False): # Check if probs requested
                if hasattr(model, 'predict_proba'):
                    try:
                        y_prob = model.predict_proba(X_pred)
                        logger.info("Probabilities calculated.")
                    except NotImplementedError:
                         logger.warning(f"Model {model_name} does not support predict_proba, probabilities will not be included.")
                    except Exception as e:
                         logger.warning(f"Error calling predict_proba for {model_name}: {e}")
                else:
                     logger.warning(f"Model {model_name} does not have predict_proba method.")

        except Exception as e:
            logger.error(f"Error during prediction with model {model_name}: {e}", exc_info=True)
            return None

        # 5. Format Output
        # Include original identifiers if possible, using the potentially processed df_to_use
        id_cols = [col for col in ['domain_id', 'chain_id', 'resid', 'icode', 'resname'] if col in df_to_use.columns]
        result_df = df_to_use[id_cols].reset_index(drop=True).copy()
        result_df['predicted_class'] = y_pred

        if y_prob is not None:
             prob_cols = [f'prob_class_{i}' for i in range(y_prob.shape[1])]
             prob_df = pd.DataFrame(y_prob, columns=prob_cols)
             result_df = pd.concat([result_df, prob_df], axis=1)

        # 6. Save or Return Results
        if output_path:
            try:
                ensure_dir(os.path.dirname(output_path))
                result_df.to_csv(output_path, index=False, float_format='%.4f')
                logger.info(f"Predictions saved to: {output_path}")
                return None # Return None when saving to file
            except Exception as e:
                 logger.error(f"Failed to save predictions to {output_path}: {e}", exc_info=True)
                 return result_df # Return df even if saving failed
        else:
             logger.info("Prediction finished. Returning DataFrame.")
             return result_df


    @timer
    def analyze_rmsf_distribution(self, input_data_path: str, output_plot_path: Optional[str] = None):
        """Analyze RMSF distribution and visualize binning."""
        logger.info("--- Analyzing RMSF Distribution ---")
        if not _plotting_available:
             logger.warning("Plotting libraries not available. Skipping RMSF distribution analysis plot.")
             return

        # 1. Load RMSF data
        target_col = self.config['dataset']['target']
        try:
            # Find the actual file path using the pattern or direct path
            data_dir = self.config['paths'].get('data_dir')
            actual_path = input_data_path
            if not os.path.isabs(input_data_path) and data_dir and not os.path.exists(input_data_path):
                 potential_path = os.path.join(data_dir, input_data_path)
                 if os.path.exists(potential_path):
                      actual_path = potential_path
                 else: # Try finding via pattern if it wasn't a direct path
                      found = find_data_file(data_dir, input_data_path)
                      if found: actual_path = found
                      else: raise FileNotFoundError(f"Cannot find input data: {input_data_path}")

            rmsf_series = load_rmsf_data(actual_path, target_column=target_col)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to load RMSF data for analysis: {e}")
            return

        # 2. Get Binner (either from instance or load)
        if not self.binner:
            binner_path = os.path.join(self.config['paths']['models_dir'], 'binner.joblib')
            if os.path.exists(binner_path):
                try:
                    self.binner = BaseBinner.load(binner_path, config=self.config)
                    logger.info(f"Loaded binner ({self.binner.__class__.__name__}) for analysis.")
                except Exception as e:
                    logger.error(f"Failed to load binner from {binner_path}: {e}. Cannot show boundaries.")
                    self.binner = None # Ensure binner is None if load failed
            else:
                logger.warning(f"Binner file not found at {binner_path}. Plotting distribution without bin boundaries.")
                self.binner = None

        # 3. Plot Distribution
        if not output_plot_path:
            output_plot_path = os.path.join(self.config['paths']['output_dir'], 'rmsf_distribution_analysis.png')

        boundaries = self.binner.get_boundaries() if self.binner else None
        try:
            plot_bin_distribution(
                rmsf_series.values,
                boundaries=boundaries, # Pass boundaries if available
                config=self.config,
                output_path=output_plot_path,
                title="RMSF Distribution Analysis" + (f" ({self.binner.__class__.__name__} Bins)" if self.binner else "")
            )
        except Exception as e:
            logger.error(f"Failed to plot RMSF distribution: {e}", exc_info=True)


    @timer
    def process_pdb(self, pdb_input: str, model_name: Optional[str] = None, output_prefix: Optional[str] = None):
        """
        Process a PDB file: Fetch/Parse -> Extract Features -> Predict -> Visualize.

        Args:
            pdb_input: PDB ID (e.g., "1AKE") or path to a local PDB file.
            model_name: Name of the model to use for prediction. Defaults to 'random_forest'.
            output_prefix: Base path and filename prefix for output files (e.g., colored PDB, PyMOL script).
                           Defaults to './pdb_output/{pdb_id}_flexibility'.
        """
        logger.info("====== Starting PDB Processing Pipeline ======")
        if not is_pdb_enabled(self.config):
             logger.error("PDB processing is disabled in configuration ('pdb.enabled=false'). Cannot proceed.")
             return

        if not model_name: model_name = "random_forest" # Default model

        # 1. Parse PDB Structure
        pdb_config = get_pdb_config(self.config)
        try:
            structure_model = parse_pdb(pdb_input, pdb_config)
            if structure_model is None: return # Error handled in parse_pdb
            structure_id = structure_model.get_parent().id # Get ID (e.g., PDB code or filename stem)
        except Exception as e:
            logger.error(f"Failed during PDB parsing stage for '{pdb_input}': {e}", exc_info=True)
            return

        # 2. Define Output Prefix
        if not output_prefix:
             output_dir = os.path.join(self.config['paths']['output_dir'], "pdb_visualizations")
             output_prefix = os.path.join(output_dir, f"{structure_id}_{model_name}_flexibility")
        ensure_dir(os.path.dirname(output_prefix))
        logger.info(f"Output prefix set to: {output_prefix}")


        # 3. Extract PDB Features
        logger.info(f"Extracting features from PDB structure: {structure_id}...")
        try:
            pdb_features_df = extract_pdb_features(structure_model, self.config)
            if pdb_features_df.empty:
                 logger.error("No features extracted from the PDB structure. Cannot proceed with prediction.")
                 return
            # Save extracted features for inspection
            features_out_path = f"{output_prefix}_extracted_features.csv"
            pdb_features_df.to_csv(features_out_path, index=False)
            logger.info(f"Extracted PDB features saved to {features_out_path}")
        except Exception as e:
            logger.error(f"Failed during PDB feature extraction for {structure_id}: {e}", exc_info=True)
            return

        # 4. Process Extracted Features (if needed - e.g., encoding, normalization, windowing)
        # The 'process_features' function expects certain columns. Ensure they match.
        # If processing is needed, apply it here. For now, assume prediction works directly on extracted features.
        # Potentially: df_processed_pdb = process_features(pdb_features_df, self.config) -> Use this for predict.
        # This depends heavily on whether models were trained *with* PDB features included.
        # Let's assume the predict function handles the necessary feature selection based on the loaded model.
        df_for_prediction = pdb_features_df

        # 5. Predict Flexibility
        logger.info(f"Predicting flexibility for {structure_id} using model '{model_name}'...")
        try:
            # Use the main predict method, ensuring it returns the DataFrame
            # Need to handle the case where predict saves to file instead of returning df
            # Modify predict or call internal prediction logic? Calling predict is cleaner.
            # We need the DataFrame back, so don't specify an output_path here.
            predictions_df = self.predict(data=df_for_prediction, model_name=model_name, output_path=None)

            if predictions_df is None or predictions_df.empty:
                 logger.error(f"Prediction failed or returned empty results for {structure_id}.")
                 return

            # Save predictions for inspection
            preds_out_path = f"{output_prefix}_predictions.csv"
            predictions_df.to_csv(preds_out_path, index=False)
            logger.info(f"Predictions saved to {preds_out_path}")

        except Exception as e:
            logger.error(f"Failed during prediction stage for {structure_id}: {e}", exc_info=True)
            return

        # 6. Generate Visualizations
        logger.info(f"Generating visualizations for {structure_id}...")
        # Ensure predictions_df has the necessary columns ('chain_id', 'resid', 'predicted_class')
        required_vis_cols = ['chain_id', 'resid', 'predicted_class']
        if not all(col in predictions_df.columns for col in required_vis_cols):
             logger.error(f"Predictions DataFrame is missing required columns for visualization: {required_vis_cols}")
             return

        # Generate PyMOL script
        try:
            pml_path = f"{output_prefix}.pml"
            pdb_filename_for_script = f"{structure_id}.pdb" # Assume standard naming if fetched
            generate_pymol_script(predictions_df, self.config, pml_path, pdb_filename=pdb_filename_for_script)
        except Exception as e:
             logger.error(f"Failed to generate PyMOL script: {e}", exc_info=True)

        # Generate colored PDB file (using B-factor)
        try:
             colored_pdb_path = f"{output_prefix}_colored.pdb"
             color_pdb_by_flexibility(structure_model, predictions_df, colored_pdb_path)
        except Exception as e:
            logger.error(f"Failed to generate colored PDB file: {e}", exc_info=True)


        logger.info(f"====== PDB Processing Finished for {structure_id} ======")


    @timer
    def run_temperature_comparison(self, model_name: Optional[str] = None):
        """Runs the temperature comparison analysis script."""
        logger.info("====== Starting Temperature Comparison Analysis ======")
        try:
             run_temperature_comparison_analysis(self.config, model_name)
        except Exception as e:
            logger.error(f"Temperature comparison analysis failed: {e}", exc_info=True)
        logger.info("====== Temperature Comparison Analysis Finished ======")

    # Potential method for the 'visualize' command (if needed beyond evaluation plots)
    def visualize_results(self, predictions_path: str, output_dir: Optional[str] = None):
         """Generates standalone visualizations from prediction results."""
         logger.info("--- Generating Standalone Visualizations ---")
         if not _plotting_available:
              logger.error("Plotting libraries not available. Cannot generate visualizations.")
              return

         if output_dir is None:
              output_dir = self.config['paths']['output_dir']
         ensure_dir(output_dir)

         try:
              pred_df = pd.read_csv(predictions_path)
              if 'predicted_class' not in pred_df.columns:
                   raise ValueError("Predictions file must contain 'predicted_class' column.")

              # Example: Plot class distribution from predictions
              dist_plot_path = os.path.join(output_dir, f"{os.path.basename(predictions_path).split('.')[0]}_class_distribution.png")
              plot_class_distribution(pred_df['predicted_class'], self.config, dist_plot_path, title="Predicted Class Distribution")

              # Add other visualizations based on prediction file content here...
              # e.g., plot confidence if probabilities are present

              logger.info(f"Standalone visualizations saved in {output_dir}")

         except FileNotFoundError:
              logger.error(f"Predictions file not found: {predictions_path}")
         except ValueError as ve:
              logger.error(f"Error processing predictions file for visualization: {ve}")
         except Exception as e:
              logger.error(f"Failed to generate standalone visualizations: {e}", exc_info=True)