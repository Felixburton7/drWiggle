import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from abc import ABC, abstractmethod
import joblib
import logging
import warnings
from typing import List, Union, Dict, Any, Optional, Type
import os

from drwiggle.config import get_binning_config

logger = logging.getLogger(__name__)

class BaseBinner(ABC):
    """Abstract base class for RMSF binning methods."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Extract binning-specific config section
        self.binning_config = get_binning_config(config)
        self.num_classes = self.binning_config.get('num_classes', 5)
        self.boundaries: Optional[List[float]] = None
        self._fitted: bool = False
        self._bin_centers: Optional[np.ndarray] = None

    @abstractmethod
    def fit(self, rmsf_values: np.ndarray):
        """
        Calculate bin boundaries from the provided RMSF values.
        Must set self.boundaries and self._fitted = True.

        Args:
            rmsf_values: 1D NumPy array of continuous RMSF values.
        """
        pass

    def transform(self, rmsf_values: np.ndarray) -> np.ndarray:
        """
        Convert continuous RMSF values to discrete class indices (0 to num_classes-1).

        Args:
            rmsf_values: 1D NumPy array of continuous RMSF values.

        Returns:
            1D NumPy array of integer class labels.

        Raises:
            RuntimeError: If the binner has not been fitted.
            ValueError: If input array is not 1D.
        """
        if not self._fitted or self.boundaries is None:
            raise RuntimeError("Binner must be fitted before transforming data.")
        if rmsf_values.ndim > 1:
             # Allow (n, 1) shape, flatten it
             if rmsf_values.ndim == 2 and rmsf_values.shape[1] == 1:
                 rmsf_values = rmsf_values.flatten()
             else:
                 raise ValueError(f"Input rmsf_values for transform must be 1D, but got shape {rmsf_values.shape}.")

        # Ensure boundaries are monotonically increasing
        if not all(self.boundaries[i] <= self.boundaries[i+1] for i in range(len(self.boundaries)-1)):
             logger.error(f"Boundaries are not monotonically increasing: {self.boundaries}. Check fitting logic.")
             # Attempt to sort? Or raise error? Sorting might hide issues.
             self.boundaries = sorted(list(set(self.boundaries))) # Ensure unique and sorted as safeguard
             if len(self.boundaries) != self.num_classes + 1:
                 raise ValueError(f"Corrected boundaries length mismatch after sorting ({len(self.boundaries)} vs {self.num_classes+1}).")


        # Use pandas.cut for efficient binning
        # Labels are 0 to num_classes-1
        # right=False: bins are [left, right) - includes left edge, excludes right edge.
        # The last bin implicitly includes the right edge because the final boundary is +inf.
        # include_lowest=True: ensures the minimum value is included in the first bin ([min_val, boundary_1)).
        labels = pd.cut(rmsf_values, bins=self.boundaries, labels=False,
                        right=False, include_lowest=True,
                        duplicates='drop') # Handle duplicate boundary edges if they occur

        # Check for NaNs which might occur if a value falls exactly on the upper boundary
        # when right=False and duplicates='drop' hasn't handled it perfectly, or if values
        # are outside the explicit min/max if (-inf, inf) weren't used.
        nan_mask = np.isnan(labels)
        if nan_mask.any():
             nan_indices = np.where(nan_mask)[0]
             # Assign NaNs to the highest class index
             highest_class_index = self.num_classes - 1
             labels[nan_mask] = highest_class_index
             logger.warning(f"Found {len(nan_indices)} NaN labels after binning (potentially values exactly on boundaries or outside range)."
                            f" Assigning them to the highest class ({highest_class_index}). "
                            f"Problematic values (first 5): {rmsf_values[nan_indices][:5]}")

        # Ensure output is integer type
        return labels.astype(int)

    def fit_transform(self, rmsf_values: np.ndarray) -> np.ndarray:
        """Fit the binner and then transform the RMSF values."""
        self.fit(rmsf_values)
        return self.transform(rmsf_values)

    def _calculate_bin_centers(self):
        """Calculate representative center for each bin."""
        if self.boundaries is None: return None
        centers = []
        for i in range(self.num_classes):
            lower = self.boundaries[i]
            upper = self.boundaries[i+1]
            if np.isneginf(lower) and np.isposinf(upper): center = 0.0 # Should not happen
            elif np.isneginf(lower): center = upper * 0.9 # Estimate slightly below upper bound
            elif np.isposinf(upper): center = lower * 1.1 # Estimate slightly above lower bound
            else: center = (lower + upper) / 2.0
            centers.append(center)
        self._bin_centers = np.array(centers)


    def inverse_transform(self, class_indices: np.ndarray) -> np.ndarray:
        """
        Convert class indices back to representative RMSF values (bin centers).

        Args:
            class_indices: 1D NumPy array of integer class labels.

        Returns:
            1D NumPy array of representative RMSF values.
        """
        if not self._fitted or self.boundaries is None:
            raise RuntimeError("Binner must be fitted before inverse transforming data.")
        if self._bin_centers is None:
             self._calculate_bin_centers()
             if self._bin_centers is None: # Still None after calculation attempt
                  raise RuntimeError("Could not calculate bin centers.")

        if len(self._bin_centers) != self.num_classes:
            raise ValueError(f"Mismatch between number of bin centers ({len(self._bin_centers)}) and num_classes ({self.num_classes}).")

        # Map indices to centers
        try:
             representative_values = self._bin_centers[class_indices]
             return representative_values
        except IndexError:
             logger.error(f"Class indices out of bounds. Max index: {np.max(class_indices)}, Num centers: {len(self._bin_centers)}")
             raise

    def get_boundaries(self) -> Optional[List[float]]:
        """Return the calculated bin boundaries."""
        return self.boundaries

    def save(self, path: str):
        """Save the fitted binner state (including boundaries and config) to a file."""
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted binner.")
        state = {
            'boundaries': self.boundaries,
            'num_classes': self.num_classes,
            'binning_config': self.binning_config, # Save specific config section used
            'fitted': self._fitted,
            'class_name': self.__class__.__name__, # Store class name for loading
            'bin_centers': self._bin_centers # Save calculated centers
        }
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True) # Ensure directory exists
            joblib.dump(state, path)
            logger.info(f"Binner state ({self.__class__.__name__}) saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save binner state to {path}: {e}")
            raise

    @classmethod
    def load(cls, path: str, config: Optional[Dict[str, Any]] = None) -> 'BaseBinner':
        """
        Load a fitted binner state from a file.

        Args:
            path: Path to the saved binner file.
            config: Optional current main config dictionary. If provided, helps ensure
                    consistency or allows using current config if saved one is minimal.

        Returns:
            A loaded instance of the appropriate BaseBinner subclass.
        """
        try:
            state = joblib.load(path)
            logger.info(f"Loading binner state from {path}")

            class_name = state.get('class_name')
            if not class_name: raise ValueError("Saved state missing 'class_name'.")

            # Find the correct class type from the registry
            binner_cls = _BINNER_REGISTRY.get(class_name)
            if not binner_cls:
                 raise ValueError(f"Cannot find binner class '{class_name}' in registry for loading.")

            # Use the config stored in the state file primarily
            saved_config = state.get('config') # Older saves might have full config
            saved_binning_config = state.get('binning_config') # Newer saves have specific section

            if saved_binning_config:
                 # Construct a minimal config dict needed for re-instantiation
                 rehydration_config = {'binning': saved_binning_config}
            elif saved_config:
                 logger.warning("Loading binner saved with older format (full config). Using saved config.")
                 rehydration_config = saved_config
            elif config: # Use provided config as fallback if nothing useful saved
                 logger.warning("No config found in saved binner state. Using provided runtime config.")
                 rehydration_config = config
            else: # Cannot reinstantiate without config
                 raise ValueError("Cannot load binner: No configuration found in saved state or provided.")

            # Re-instantiate the class
            instance = binner_cls(config=rehydration_config)

            # Restore state
            instance.boundaries = state.get('boundaries')
            instance.num_classes = state.get('num_classes', instance.num_classes) # Use loaded if available
            instance._fitted = state.get('fitted', False)
            instance._bin_centers = state.get('bin_centers')

            if not instance._fitted or instance.boundaries is None:
                 logger.warning(f"Loaded binner state from {path} appears incomplete or unfitted.")

            return instance
        except FileNotFoundError:
             logger.error(f"Binner state file not found: {path}")
             raise
        except Exception as e:
            logger.error(f"Failed to load binner state from {path}: {e}")
            raise


class KMeansBinner(BaseBinner):
    """Bins RMSF values using K-means clustering on the 1D data."""

    def fit(self, rmsf_values: np.ndarray):
        """Calculate bin boundaries using K-means."""
        if rmsf_values.ndim > 1:
             if rmsf_values.ndim == 2 and rmsf_values.shape[1] == 1:
                 rmsf_values = rmsf_values.flatten()
             else:
                 raise ValueError(f"Input rmsf_values for KMeansBinner fit must be 1D, but got shape {rmsf_values.shape}.")

        if len(np.unique(rmsf_values)) < self.num_classes:
             logger.warning(f"Number of unique RMSF values ({len(np.unique(rmsf_values))}) is less than num_classes ({self.num_classes}). KMeans may produce fewer clusters or fail.")

        rmsf_reshaped = rmsf_values.reshape(-1, 1) # KMeans expects 2D array

        kmeans_params = self.binning_config.get('kmeans', {})
        kmeans = KMeans(
            n_clusters=self.num_classes,
            random_state=kmeans_params.get('random_state', 42),
            max_iter=kmeans_params.get('max_iter', 300),
            n_init=kmeans_params.get('n_init', 10),
            init='k-means++' # Default robust initialization
        )

        logger.info(f"Fitting KMeans with k={self.num_classes}...")
        start_time = pd.Timestamp.now()
        try:
             with warnings.catch_warnings():
                 # Suppress ConvergenceWarning which is common with multiple inits
                 warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.cluster._kmeans")
                 kmeans.fit(rmsf_reshaped)
        except Exception as e:
             logger.error(f"KMeans fitting failed: {e}")
             logger.warning("Falling back to Quantile binning due to KMeans error.")
             # Fallback logic
             quantile_binner = QuantileBinner(self.config)
             quantile_binner.fit(rmsf_values)
             self.boundaries = quantile_binner.boundaries
             self._fitted = quantile_binner._fitted
             return # Exit after fallback


        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(f"KMeans fitting complete in {elapsed:.2f}s. Inertia: {kmeans.inertia_:.2f}")

        centers = sorted(kmeans.cluster_centers_.flatten())
        actual_num_clusters = len(centers)
        logger.debug(f"KMeans cluster centers found ({actual_num_clusters}): {np.round(centers, 3)}")

        if actual_num_clusters < self.num_classes:
            logger.warning(f"KMeans found only {actual_num_clusters} clusters, less than requested ({self.num_classes}). Bin boundaries might merge classes.")
            # If significantly fewer, might indicate issues with data or k choice.
            if actual_num_clusters < 2:
                 logger.error("KMeans found less than 2 clusters. Cannot define meaningful boundaries. Falling back to Quantiles.")
                 quantile_binner = QuantileBinner(self.config)
                 quantile_binner.fit(rmsf_values)
                 self.boundaries = quantile_binner.boundaries
                 self._fitted = quantile_binner._fitted
                 return

        # Calculate boundaries as midpoints between sorted centers
        if actual_num_clusters > 1:
             boundaries_mid = [(centers[i] + centers[i+1]) / 2.0 for i in range(actual_num_clusters - 1)]
        else: # Only 1 cluster center
             # Create somewhat arbitrary boundaries around the single center
             std_dev = np.std(rmsf_values) if len(rmsf_values) > 1 else 0.1
             boundaries_mid = [centers[0] - std_dev*0.5, centers[0] + std_dev*0.5]
             # Adjust num_classes if only one center found? Or force failure? Forcing Quantile fallback is safer.
             logger.error("KMeans found only 1 cluster center. Falling back to Quantiles.")
             quantile_binner = QuantileBinner(self.config)
             quantile_binner.fit(rmsf_values)
             self.boundaries = quantile_binner.boundaries
             self._fitted = quantile_binner._fitted
             return


        # Add outer boundaries: -inf and +inf for robustness
        self.boundaries = [-np.inf] + boundaries_mid + [np.inf]

        # Ensure the number of boundaries matches num_classes + 1
        # If KMeans found fewer clusters, we might have fewer midpoints.
        # We need to pad the boundaries list. A simple way is to repeat the last finite boundary
        # or add boundaries based on std dev, but this indicates a potential issue.
        # A safer approach if boundaries are insufficient is to fallback to quantile.
        if len(self.boundaries) != self.num_classes + 1:
             logger.warning(f"Generated {len(self.boundaries)} boundaries (expected {self.num_classes + 1}) "
                            f"due to {actual_num_clusters} clusters found. "
                            f"Boundaries: {np.round(self.boundaries, 3)}. "
                            "Falling back to Quantile binning for reliable boundary count.")
             # Fallback logic
             quantile_binner = QuantileBinner(self.config)
             quantile_binner.fit(rmsf_values)
             self.boundaries = quantile_binner.boundaries

        # Final check for monotonicity
        if not all(self.boundaries[i] <= self.boundaries[i+1] for i in range(len(self.boundaries)-1)):
            logger.error(f"KMeans boundaries are not monotonic: {self.boundaries}. Falling back to Quantiles.")
            quantile_binner = QuantileBinner(self.config)
            quantile_binner.fit(rmsf_values)
            self.boundaries = quantile_binner.boundaries

        self._fitted = True
        logger.info(f"KMeansBinner fitted. Boundaries: {np.round(self.boundaries, 3)}")


class QuantileBinner(BaseBinner):
    """Bins RMSF values based on quantiles (percentiles)."""

    def fit(self, rmsf_values: np.ndarray):
        """Calculate bin boundaries using quantiles."""
        if rmsf_values.ndim > 1:
             if rmsf_values.ndim == 2 and rmsf_values.shape[1] == 1:
                 rmsf_values = rmsf_values.flatten()
             else:
                 raise ValueError(f"Input rmsf_values for QuantileBinner fit must be 1D, but got shape {rmsf_values.shape}.")

        quantile_params = self.binning_config.get('quantile', {})
        percentiles = quantile_params.get('percentiles')

        # If percentiles not provided or don't match num_classes, generate defaults
        if percentiles is None or len(percentiles) != self.num_classes + 1:
            if percentiles is not None: # Provided but wrong length
                 logger.warning(f"Number of percentiles ({len(percentiles)}) in config does not match num_classes+1 ({self.num_classes+1}). Ignoring config and generating default percentiles.")
            percentiles = np.linspace(0, 100, self.num_classes + 1)
            logger.info(f"Using default percentiles for {self.num_classes} classes: {np.round(percentiles, 1)}")
        else:
             logger.info(f"Using percentiles from config: {percentiles}")


        # Calculate boundaries using numpy.percentile
        boundaries = np.percentile(rmsf_values, percentiles)

        # Handle non-unique boundaries (can happen with discrete data or many identical values)
        unique_boundaries = np.unique(boundaries)
        if len(unique_boundaries) < len(boundaries):
            logger.warning(f"Quantile calculation resulted in non-unique boundaries ({np.round(boundaries, 3)}). "
                           f"Unique values: {np.round(unique_boundaries, 3)}. Binning might merge classes.")
            # Strategy: Keep unique values. This might reduce the effective number of classes.
            boundaries = unique_boundaries

            # If we have too few boundaries now, log an error, but proceed.
            # pd.cut with duplicates='drop' should handle this, but classes will be merged.
            if len(boundaries) <= self.num_classes: # Need at least num_classes+1 edges initially
                 logger.error(f"Only {len(boundaries)} unique boundaries found after quantile calculation, expected {self.num_classes + 1}. "
                              f"Effective number of classes will be reduced.")
                 # Cannot reliably proceed with the configured number of classes.
                 # Raising an error might be better than silently merging classes.
                 # raise ValueError(f"Could not determine {self.num_classes} distinct quantile bins.")
                 # Alternative: Adjust num_classes? For now, warn and proceed.
                 pass # Let pd.cut handle it, classes will merge

        # Ensure boundaries start at -inf and end at +inf for robustness
        # np.percentile(a, 0) gives min, np.percentile(a, 100) gives max
        # Replace first with -inf and last with +inf
        final_boundaries = [-np.inf] + boundaries[1:-1].tolist() + [np.inf]

        # If boundaries became non-unique *after* replacing with inf (only possible if min/max were equal)
        # Or if the number of boundaries is still wrong (e.g. only 2 unique values found)
        check_boundaries = sorted(list(set(b for b in final_boundaries if np.isfinite(b))))
        if len(check_boundaries) < self.num_classes -1: # Need at least N-1 finite boundaries for N classes
             logger.error(f"Insufficient distinct finite boundaries ({len(check_boundaries)}) found for {self.num_classes} classes. Check data distribution.")
             # Fallback: Create arbitrary linear spacing if quantile fails badly? Risky.
             # Let's raise an error here as the binning is likely meaningless.
             raise ValueError(f"Failed to create sufficient distinct quantile boundaries ({len(final_boundaries)} total) for {self.num_classes} classes.")

        self.boundaries = final_boundaries
        self._fitted = True
        logger.info(f"QuantileBinner fitted. Boundaries: {np.round(self.boundaries, 3)}")


# --- Binner Registry and Factory ---

_BINNER_REGISTRY: Dict[str, Type[BaseBinner]] = {
    "kmeans": KMeansBinner,
    "quantile": QuantileBinner,
}

def get_binner(config: Dict[str, Any]) -> BaseBinner:
    """
    Factory function to instantiate a binner based on the configuration.

    Args:
        config: The main configuration dictionary.

    Returns:
        An instance of the appropriate BaseBinner subclass.

    Raises:
        ValueError: If the specified binning method is unknown.
    """
    binning_config = get_binning_config(config)
    method = binning_config.get('method', 'kmeans').lower() # Default to kmeans if not specified

    binner_cls = _BINNER_REGISTRY.get(method)
    if binner_cls:
        logger.info(f"Creating binner using method: '{method}'")
        return binner_cls(config)
    else:
        raise ValueError(f"Unknown binning method: '{method}'. Available methods: {list(_BINNER_REGISTRY.keys())}")
