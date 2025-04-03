import logging
import os
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

# Configure Matplotlib backend for compatibility
import matplotlib
try:
    # Try using 'Agg' first for non-interactive environments
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    _plotting_available = True
except ImportError:
    # Fallback or warning if libraries not installed
    logging.getLogger(__name__).warning("Matplotlib or Seaborn not found. Plotting functions will be disabled. Install them (`pip install matplotlib seaborn`)")
    # Define dummy plt, sns if not available to avoid runtime errors later
    class DummyPlt:
        def subplots(self, *args, **kwargs): return None, DummyAx()
        def close(self, *args, **kwargs): pass
        def __getattr__(self, name): return lambda *args, **kwargs: None # Dummy any other plt calls
    class DummyAx:
        def __getattr__(self, name): return lambda *args, **kwargs: None
    plt = DummyPlt()
    sns = None # Assign None to sns
    _plotting_available = False
except Exception as e:
    # Handle other potential backend errors
     logging.getLogger(__name__).warning(f"Error setting up Matplotlib backend 'Agg': {e}. Plotting might fail. Trying default backend.")
     try:
         import matplotlib.pyplot as plt
         import seaborn as sns
         _plotting_available = True
     except ImportError:
        logging.getLogger(__name__).warning("Matplotlib or Seaborn not found after backend error. Plotting disabled.")
        plt = DummyPlt()
        sns = None
        _plotting_available = False


from drwiggle.utils.helpers import ensure_dir
from drwiggle.config import get_visualization_colors, get_class_names

logger = logging.getLogger(__name__)

# --- Plot Saving Helper ---

def _save_plot(figure: Optional[matplotlib.figure.Figure], output_path: str, dpi: int = 150):
    """Helper function to save a Matplotlib plot and close the figure."""
    if not _plotting_available or figure is None:
        logger.warning(f"Plotting libraries unavailable or figure invalid. Cannot save plot to {output_path}")
        if figure: plt.close(figure) # Close even if not saved
        return

    try:
        ensure_dir(os.path.dirname(output_path))
        # Use tight_layout with padding to prevent labels overlapping edges
        figure.tight_layout(pad=1.1)
        figure.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save plot to {output_path}: {e}", exc_info=True)
    finally:
        # Ensure the figure is closed to free memory, regardless of saving success
        plt.close(figure)

# --- Specific Plotting Functions ---

def plot_bin_distribution(
    rmsf_values: Union[np.ndarray, pd.Series],
    boundaries: List[float],
    config: Dict[str, Any],
    output_path: str,
    num_bins_hist: int = 50,
    title: Optional[str] = None
):
    """Plots the RMSF distribution histogram with bin boundaries marked."""
    if not _plotting_available: return
    if boundaries is None or len(boundaries) < 2:
         logger.warning("Cannot plot bin distribution: Invalid boundaries provided.")
         return
    if isinstance(rmsf_values, pd.Series): rmsf_values = rmsf_values.values

    # Filter out NaNs or Infs from RMSF values before plotting
    finite_rmsf = rmsf_values[np.isfinite(rmsf_values)]
    if len(finite_rmsf) == 0:
         logger.warning("No finite RMSF values to plot for distribution.")
         return
    if len(finite_rmsf) < len(rmsf_values):
         logger.warning(f"Plotting RMSF distribution using {len(finite_rmsf)} finite values (excluded {len(rmsf_values) - len(finite_rmsf)} non-finite).")


    fig, ax = plt.subplots(figsize=(10, 6))
    # Use seaborn's histplot for potentially nicer aesthetics and KDE integration
    sns.histplot(finite_rmsf, bins=num_bins_hist, kde=True, stat="density", alpha=0.7, label='RMSF Distribution', ax=ax)

    # Add vertical lines for boundaries
    colors_map = get_visualization_colors(config)
    num_classes = len(boundaries) - 1
    # Use a default colormap if config colors don't match num_classes or are missing
    default_cmap = plt.cm.viridis
    class_colors = [colors_map.get(i, default_cmap(i / num_classes)) for i in range(num_classes)]


    for i, bound in enumerate(boundaries):
        if np.isfinite(bound): # Skip infinite boundaries
             # Assign color based on the *bin to the right* of the boundary line
             color_idx = min(i, num_classes - 1) # Use color of the bin starting at this boundary
             line_color = class_colors[color_idx]
             ax.axvline(bound, color=line_color, linestyle='--', lw=1.5,
                        label=f'Boundary {i}' if (i==0 and np.isfinite(boundaries[0])) or (i==len(boundaries)-1 and np.isfinite(boundaries[-1])) else None) # Label first/last finite maybe?

    ax.set_xlabel("RMSF Value (Å)") # Assuming Angstroms
    ax.set_ylabel("Density")
    plot_title = title or "RMSF Distribution and Class Boundaries"
    ax.set_title(plot_title)
    # Improve legend handling - maybe only add one entry for boundaries
    # handles, labels = ax.get_legend_handles_labels()
    # if handles: ax.legend(handles=handles, labels=labels, fontsize='small', loc='best')
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    # Set sensible x-limits based on data range
    data_min, data_max = np.min(finite_rmsf), np.max(finite_rmsf)
    data_std = np.std(finite_rmsf)
    ax.set_xlim(left=max(0, data_min - data_std*0.2),
                right=data_max + data_std*0.2)


    _save_plot(fig, output_path)

def plot_confusion_matrix(
    cm_df: pd.DataFrame, # Confusion matrix as DataFrame from utils.metrics
    output_path: str,
    normalize: bool = True,
    title: Optional[str] = None
):
    """Plots the confusion matrix from a DataFrame."""
    if not _plotting_available or sns is None: return
    if cm_df is None or cm_df.empty:
         logger.warning("Cannot plot confusion matrix: Input DataFrame is None or empty.")
         return

    cm_array = cm_df.values
    class_names = cm_df.columns.tolist() # Get names from columns

    if normalize:
        # Normalize by row (true label) to get recall per class on diagonal
        cm_sum = cm_array.sum(axis=1)[:, np.newaxis]
        # Avoid division by zero for classes with no samples
        with np.errstate(divide='ignore', invalid='ignore'):
             cm_norm = cm_array.astype('float') / cm_sum
        cm_norm[np.isnan(cm_norm)] = 0 # Set NaNs (from 0/0) to 0
        data_to_plot = cm_norm
        fmt = '.2f'
        plot_title = title or "Normalized Confusion Matrix (Recall)"
    else:
        data_to_plot = cm_array
        fmt = 'd'
        plot_title = title or 'Confusion Matrix (Counts)'

    fig, ax = plt.subplots(figsize=(max(6, len(class_names)*0.8), max(5, len(class_names)*0.7)))
    sns.heatmap(data_to_plot, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=.5, linecolor='lightgray', annot_kws={"size": 10}) # Adjust annotation size

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(plot_title, fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

    _save_plot(fig, output_path)


def plot_feature_importance(
    importances: Dict[str, float], # Dict {feature_name: score}
    output_path: str,
    top_n: int = 25,
    title: Optional[str] = None,
    model_name: Optional[str] = None
):
    """Plots the top N feature importances from a dictionary."""
    if not _plotting_available or sns is None: return
    if not importances:
        logger.warning("No feature importances provided to plot.")
        return

    # Create DataFrame and sort
    importance_df = pd.DataFrame(list(importances.items()), columns=['feature', 'importance'])
    # Filter out near-zero-importance features before selecting top N
    importance_df = importance_df[importance_df['importance'] > 1e-6] # Threshold near zero
    importance_df = importance_df.sort_values(by='importance', ascending=False).head(top_n)

    if importance_df.empty:
         logger.warning("No features with importance > 0 found to plot.")
         return

    # Plotting
    fig, ax = plt.subplots(figsize=(10, max(6, len(importance_df) * 0.35))) # Adjust height
    sns.barplot(x='importance', y='feature', data=importance_df, ax=ax, palette='viridis_r') # Reverse viridis

    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_ylabel("Feature Name", fontsize=12)
    plot_title = title or f"Top {len(importance_df)} Feature Importances"
    if model_name: plot_title += f" ({model_name})"
    ax.set_title(plot_title, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(axis='x', linestyle=':', alpha=0.6)

    _save_plot(fig, output_path)

def plot_class_distribution(
    class_labels: Union[np.ndarray, pd.Series],
    config: Dict[str, Any],
    output_path: str,
    title: Optional[str] = None
):
    """Plots the distribution of predicted or actual classes."""
    if not _plotting_available or sns is None: return
    if isinstance(class_labels, pd.Series): class_labels = class_labels.values
    if class_labels is None or len(class_labels) == 0:
        logger.warning("No class labels provided for plotting distribution.")
        return


    class_names_map = get_class_names(config)
    num_classes = len(class_names_map) if class_names_map else (np.max(class_labels) + 1 if len(class_labels) > 0 else 0)
    if num_classes == 0:
        logger.warning("Cannot plot class distribution: Number of classes is zero.")
        return
    class_names = [class_names_map.get(i, f"Class_{i}") for i in range(num_classes)]

    unique_classes, counts = np.unique(class_labels, return_counts=True)
    class_counts = dict(zip(unique_classes, counts))

    # Ensure all classes are represented, even if count is 0
    plot_data = pd.DataFrame({
        'class_index': range(num_classes),
        'class_name': class_names,
        'count': [class_counts.get(i, 0) for i in range(num_classes)]
    })
    total_count = plot_data['count'].sum()
    plot_data['percentage'] = (plot_data['count'] / total_count) * 100 if total_count > 0 else 0

    fig, ax = plt.subplots(figsize=(max(8, num_classes * 1.2), 6)) # Adjusted size
    colors_map = get_visualization_colors(config)
    colors_list = [colors_map.get(i, None) for i in range(num_classes)] if colors_map else None
    # Use default palette if colors not fully defined
    palette = colors_list if colors_list and all(colors_list) else sns.color_palette()

    bar_plot = sns.barplot(x='class_name', y='count', data=plot_data, ax=ax, palette=palette)

    # Add percentage labels on top of bars
    for index, row in plot_data.iterrows():
        if row['count'] > 0: # Only label bars with counts
             ax.text(index, row['count'], f"{row['percentage']:.1f}%",
                     color='black', ha="center", va='bottom', fontsize=9)

    ax.set_xlabel("Flexibility Class", fontsize=12)
    ax.set_ylabel("Number of Residues", fontsize=12)
    plot_title = title or "Class Distribution"
    ax.set_title(plot_title, fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    # Adjust y-limit to make space for text labels
    ax.set_ylim(top=ax.get_ylim()[1] * 1.1)

    _save_plot(fig, output_path)

# --- Temperature Comparison Plots ---

def plot_metric_vs_temperature(
     metrics_df: pd.DataFrame, # Should contain 'temperature', 'model', and metric columns
     metric: str,
     output_path: str,
     title: Optional[str] = None
):
    """Plots a specific metric against temperature for different models."""
    if not _plotting_available or sns is None: return
    if metrics_df is None or metrics_df.empty or metric not in metrics_df.columns:
        logger.warning(f"Cannot plot metric '{metric}' vs temperature: DataFrame empty or metric column missing.")
        return

    # Ensure temperature is numeric, handle non-numeric gracefully (like 'average')
    metrics_df['temperature_num'] = pd.to_numeric(metrics_df['temperature'], errors='coerce')
    plot_df = metrics_df.dropna(subset=['temperature_num', metric]).copy()
    plot_df = plot_df.sort_values(by=['model', 'temperature_num'])

    if plot_df.empty:
        logger.warning(f"No valid data points found for metric '{metric}' vs temperature plot.")
        return

    num_models = plot_df['model'].nunique()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use Seaborn for potentially nicer plotting with hue for models
    sns.lineplot(data=plot_df, x='temperature_num', y=metric, hue='model',
                 marker='o', ax=ax, markersize=7, legend='full')

    ax.set_xlabel("Temperature (K)", fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12) # Nicer label
    plot_title = title or f"{ax.get_ylabel()} vs Temperature"
    ax.set_title(plot_title, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Adjust legend position if many models
    if num_models > 5:
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    else:
        ax.legend(title='Model', fontsize=10)

    ax.grid(True, linestyle=':', alpha=0.7)

    _save_plot(fig, output_path)


def plot_transition_matrix(
     transition_matrix: pd.DataFrame, # Rows=T1, Cols=T2, Values=Count/Prob
     t1_name: str,
     t2_name: str,
     output_path: str,
     normalize: bool = True,
     title: Optional[str] = None
):
     """Plots a class transition matrix between two temperatures."""
     if not _plotting_available or sns is None: return
     if transition_matrix is None or transition_matrix.empty:
        logger.warning(f"Cannot plot transition matrix from {t1_name} to {t2_name}: Matrix is None or empty.")
        return

     matrix_data = transition_matrix.values
     class_names = transition_matrix.columns.tolist() # Assume row/col names match

     if normalize:
          # Normalize by row (T1 class) -> probability of transitioning to T2 class
          row_sums = matrix_data.sum(axis=1)[:, np.newaxis]
          with np.errstate(divide='ignore', invalid='ignore'):
               matrix_norm = matrix_data.astype('float') / row_sums
          matrix_norm[np.isnan(matrix_norm)] = 0 # Handle division by zero for rows with no samples
          data_to_plot = matrix_norm
          fmt = '.2f'
          plot_title = title or f"Normalized Class Transition Probability ({t1_name} K -> {t2_name} K)"
          cmap = "viridis" # Use a sequential colormap for probabilities
     else:
          data_to_plot = matrix_data
          fmt = 'd'
          plot_title = title or f"Class Transition Counts ({t1_name} K -> {t2_name} K)"
          cmap = "Blues" # Use Blues for counts

     fig, ax = plt.subplots(figsize=(max(6, len(class_names)*0.9), max(5, len(class_names)*0.8)))
     sns.heatmap(data_to_plot, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                 xticklabels=class_names, yticklabels=class_names,
                 linewidths=.5, linecolor='lightgray', annot_kws={"size": 10})

     ax.set_xlabel(f"Predicted Class at {t2_name} K", fontsize=12)
     ax.set_ylabel(f"Predicted Class at {t1_name} K", fontsize=12)
     ax.set_title(plot_title, fontsize=14)
     plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
     plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)

     _save_plot(fig, output_path)
