import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    mean_absolute_error # For ordinal error
)
# Weighted kappa requires careful handling due to potential version differences
try:
    # Prefer newer signature if available
    from sklearn.metrics import cohen_kappa_score as calculate_weighted_kappa
    _kappa_supports_weights = True
except ImportError:
    # Fallback for older versions might exist but is less standard
    logger = logging.getLogger(__name__)
    logger.warning("Could not import weighted kappa calculation logic from sklearn.metrics. Weighted Kappa might be unavailable.")
    _kappa_supports_weights = False
    def calculate_weighted_kappa(*args, **kwargs): return np.nan

logger = logging.getLogger(__name__)

def calculate_ordinal_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the Mean Absolute Error between class indices."""
    if y_true.shape != y_pred.shape:
        logger.error(f"Shape mismatch for ordinal error: y_true={y_true.shape}, y_pred={y_pred.shape}")
        # Return NaN or raise error? Returning NaN for now.
        return np.nan
    if y_true.ndim != 1:
        logger.warning(f"Ordinal error expects 1D arrays, got y_true={y_true.ndim}D, y_pred={y_pred.ndim}D. Flattening.")
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
    try:
        return mean_absolute_error(y_true, y_pred)
    except Exception as e:
         logger.error(f"Failed to calculate ordinal error (MAE): {e}", exc_info=True)
         return np.nan


def evaluate_classification(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    config: Dict[str, Any],
    y_prob: Optional[Union[np.ndarray, pd.DataFrame]] = None, # Add probabilities if available
    model_name: Optional[str] = "Unknown Model" # For logging context
) -> Dict[str, float]:
    """
    Evaluate classification performance using metrics specified in the config.

    Args:
        y_true: True class labels (n_samples,).
        y_pred: Predicted class labels (n_samples,).
        config: Configuration dictionary containing evaluation settings.
        y_prob: Predicted probabilities (n_samples, n_classes), optional.
        model_name: Name of the model being evaluated (for logging).

    Returns:
        Dictionary containing calculated metric values (floats). Returns NaN for failed metrics.
    """
    metrics_config = config.get("evaluation", {}).get("metrics", {})
    class_names_map = config.get("evaluation", {}).get("class_names", {})
    num_classes = len(class_names_map) if class_names_map else 0

    # Infer num_classes if not explicitly available
    if num_classes == 0 and (y_true is not None and len(y_true) > 0) and (y_pred is not None and len(y_pred) > 0):
        try:
            max_label = int(max(np.max(y_true), np.max(y_pred)))
            num_classes = max_label + 1
            logger.warning(f"Number of classes not found in config, inferred as {num_classes} from data.")
        except Exception:
            logger.error("Could not infer number of classes from y_true/y_pred.")
            num_classes = 0 # Set back to 0 if inference fails


    class_labels = list(range(num_classes)) if num_classes > 0 else []

    results: Dict[str, float] = {} # Store results here

    # Convert inputs to numpy arrays if they are pandas Series/DataFrames
    if isinstance(y_true, pd.Series): y_true = y_true.values
    if isinstance(y_pred, pd.Series): y_pred = y_pred.values
    if isinstance(y_prob, pd.DataFrame): y_prob = y_prob.values

    # Basic input validation
    if y_true is None or y_pred is None:
         logger.error(f"Missing y_true or y_pred for evaluation ({model_name}). Cannot calculate metrics.")
         return {metric: np.nan for metric in metrics_config if metrics_config.get(metric)}
    if y_true.shape != y_pred.shape:
        logger.error(f"Shape mismatch for evaluation ({model_name}): y_true={y_true.shape}, y_pred={y_pred.shape}")
        return {metric: np.nan for metric in metrics_config if metrics_config.get(metric)}
    if len(y_true) == 0:
         logger.warning(f"Evaluation arrays for {model_name} are empty. Returning NaN for all metrics.")
         return {metric: np.nan for metric in metrics_config if metrics_config.get(metric)}
    if num_classes == 0:
         logger.error(f"Cannot determine number of classes for evaluation ({model_name}). Metrics requiring labels will fail.")
         # Allow metrics like accuracy that don't strictly need num_classes?

    # --- Calculate Enabled Metrics ---
    logger.info(f"Calculating evaluation metrics for {model_name}...")
    try:
        if metrics_config.get("accuracy"):
            try:
                results["accuracy"] = accuracy_score(y_true, y_pred)
            except Exception as e:
                 logger.warning(f"Failed to calculate accuracy: {e}")
                 results["accuracy"] = np.nan

        if metrics_config.get("balanced_accuracy"):
             try:
                 # Requires labels if not all classes are present in y_true/y_pred
                 results["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
             except Exception as e:
                 logger.warning(f"Failed to calculate balanced_accuracy: {e}")
                 results["balanced_accuracy"] = np.nan

        # Precision, Recall, F1
        prf1_metrics = {"precision", "recall", "f1"}
        if any(metrics_config.get(m) for m in prf1_metrics):
            try:
                # Calculate macro and weighted averages
                # Pass labels to ensure calculation considers all potential classes
                p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='macro', zero_division=0, labels=class_labels if class_labels else None
                )
                p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='weighted', zero_division=0, labels=class_labels if class_labels else None
                )

                if metrics_config.get("precision"):
                     results["precision_macro"] = p_macro
                     results["precision_weighted"] = p_weighted
                if metrics_config.get("recall"):
                     results["recall_macro"] = r_macro
                     results["recall_weighted"] = r_weighted
                if metrics_config.get("f1"):
                     results["f1_macro"] = f1_macro
                     results["f1_weighted"] = f1_weighted

            except Exception as e:
                 logger.warning(f"Failed to calculate precision/recall/f1 scores: {e}")
                 if metrics_config.get("precision"): results["precision_macro"] = results["precision_weighted"] = np.nan
                 if metrics_config.get("recall"): results["recall_macro"] = results["recall_weighted"] = np.nan
                 if metrics_config.get("f1"): results["f1_macro"] = results["f1_weighted"] = np.nan

        if metrics_config.get("cohen_kappa"):
            try:
                results["cohen_kappa"] = cohen_kappa_score(y_true, y_pred, labels=class_labels if class_labels else None)
            except Exception as e:
                 logger.warning(f"Failed to calculate cohen_kappa: {e}")
                 results["cohen_kappa"] = np.nan

        if metrics_config.get("weighted_kappa"):
             if _kappa_supports_weights:
                 try:
                     # Use quadratic weights suitable for ordinal classes
                     results["weighted_kappa"] = calculate_weighted_kappa(y_true, y_pred, weights='quadratic', labels=class_labels if class_labels else None)
                 except Exception as e:
                      logger.warning(f"Failed to calculate weighted_kappa (quadratic): {e}")
                      results["weighted_kappa"] = np.nan
             else:
                  logger.warning("Weighted kappa calculation skipped as it's not supported or import failed.")
                  results["weighted_kappa"] = np.nan


        if metrics_config.get("ordinal_error"):
            results["ordinal_error"] = calculate_ordinal_error(y_true, y_pred)

        # Add more metrics here if needed (e.g., AUC, LogLoss if y_prob provided)
        # Example: Log Loss
        # if metrics_config.get("log_loss") and y_prob is not None:
        #     from sklearn.metrics import log_loss
        #     try:
        #         # Ensure y_prob has shape (n_samples, n_classes) and covers all potential classes
        #         if y_prob.shape[1] < num_classes:
        #              logger.warning(f"y_prob has fewer columns ({y_prob.shape[1]}) than classes ({num_classes}). Cannot compute log_loss accurately.")
        #              results["log_loss"] = np.nan
        #         else:
        #              # Need to ensure labels passed to log_loss match the columns of y_prob
        #              results["log_loss"] = log_loss(y_true, y_prob[:, :num_classes], labels=class_labels)
        #     except Exception as e:
        #          logger.warning(f"Failed to calculate log_loss: {e}")
        #          results["log_loss"] = np.nan


    except Exception as e:
        logger.error(f"Unexpected error during metric calculation for {model_name}: {e}", exc_info=True)
        # Ensure all requested metrics have a NaN entry if calculation failed globally
        for metric in metrics_config:
            if metrics_config.get(metric) and metric not in results:
                 results[metric] = np.nan

    # Log computed metrics
    metrics_log = ", ".join([f"{k}: {v:.4f}" for k, v in results.items() if not np.isnan(v)])
    logger.info(f"Evaluation Metrics ({model_name}): {metrics_log}")
    if any(np.isnan(v) for v in results.values()):
        nan_metrics = [k for k, v in results.items() if np.isnan(v)]
        logger.warning(f"Could not compute the following metrics for {model_name}: {nan_metrics}")


    return results

# --- Functions to generate report structures (can be saved/printed later) ---

def generate_classification_report_dict(
     y_true: Union[np.ndarray, pd.Series],
     y_pred: Union[np.ndarray, pd.Series],
     config: Dict[str, Any], # Pass config to get class names
) -> Dict[str, Any]:
     """
     Generates a classification report as a dictionary using sklearn.

     Args:
         y_true: True class labels.
         y_pred: Predicted class labels.
         config: Main configuration dictionary.

     Returns:
         Dictionary representation of the classification report. Returns empty dict on error.
     """
     if isinstance(y_true, pd.Series): y_true = y_true.values
     if isinstance(y_pred, pd.Series): y_pred = y_pred.values

     class_names_map = config.get("evaluation", {}).get("class_names", {})
     num_classes = len(class_names_map) if class_names_map else 0
     if num_classes == 0: # Infer if needed
          try:
               max_label = int(max(np.max(y_true), np.max(y_pred)))
               num_classes = max_label + 1
          except Exception: num_classes = 0

     labels = list(range(num_classes)) if num_classes > 0 else None
     target_names = [class_names_map.get(i, f"Class_{i}") for i in labels] if labels and class_names_map else None

     logger.debug(f"Generating classification report with labels={labels}, target_names={target_names}")

     try:
          # Ensure labels arg matches target_names if provided
          report = classification_report(
               y_true, y_pred,
               labels=labels,
               target_names=target_names,
               output_dict=True,
               zero_division=0
          )
          return report
     except Exception as e:
          logger.error(f"Could not generate classification report: {e}", exc_info=True)
          return {}

def generate_confusion_matrix_df(
     y_true: Union[np.ndarray, pd.Series],
     y_pred: Union[np.ndarray, pd.Series],
     config: Dict[str, Any], # Pass config to get class names
) -> Optional[pd.DataFrame]:
     """
     Calculates the confusion matrix and returns it as a pandas DataFrame.

     Args:
         y_true: True class labels.
         y_pred: Predicted class labels.
         config: Main configuration dictionary.

     Returns:
         pandas DataFrame representing the confusion matrix, or None on error.
         Rows represent True labels, Columns represent Predicted labels.
     """
     if isinstance(y_true, pd.Series): y_true = y_true.values
     if isinstance(y_pred, pd.Series): y_pred = y_pred.values

     class_names_map = config.get("evaluation", {}).get("class_names", {})
     num_classes = len(class_names_map) if class_names_map else 0
     if num_classes == 0: # Infer if needed
          try:
               max_label = int(max(np.max(y_true), np.max(y_pred)))
               num_classes = max_label + 1
          except Exception: num_classes = 0

     labels = list(range(num_classes)) if num_classes > 0 else []
     cm_labels = [class_names_map.get(i, f"Class_{i}") for i in labels] if labels and class_names_map else None
     if labels and not cm_labels: # Generate default names if needed
          cm_labels = [f"Class_{i}" for i in labels]


     if not labels:
          logger.warning("Cannot generate confusion matrix: number of classes is zero or could not be determined.")
          return None

     logger.debug(f"Generating confusion matrix with labels={labels}, class_names={cm_labels}")

     try:
          cm_array = confusion_matrix(y_true, y_pred, labels=labels)
          cm_df = pd.DataFrame(cm_array, index=cm_labels, columns=cm_labels)
          cm_df.index.name = 'True Label'
          cm_df.columns.name = 'Predicted Label'
          return cm_df
     except Exception as e:
          logger.error(f"Could not calculate confusion matrix: {e}", exc_info=True)
          return None
