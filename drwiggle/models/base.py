from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import joblib # Common choice for saving sklearn-based models
import logging
import os

from drwiggle.config import get_model_config # Assumes this helper exists

logger = logging.getLogger(__name__)

class BaseClassifier(ABC):
    """
    Abstract Base Class for all drWiggle classification models.

    Defines the common interface for training, prediction, saving, loading,
    and accessing feature importances or performing hyperparameter optimization.
    """

    def __init__(self, config: Dict[str, Any], model_name: str):
        """
        Initialize the base classifier.

        Args:
            config: The main configuration dictionary, potentially already
                    path-resolved and templated.
            model_name: The specific name of this model (e.g., 'random_forest'),
                        used to fetch its configuration subsection.
        """
        self.config = config
        self.model_name = model_name
        # Get the specific configuration for this model, merged with common settings
        self.model_config = get_model_config(config, model_name)
        self.model = None # Placeholder for the actual underlying ML model instance
        self.feature_names_in_: Optional[List[str]] = None # Store feature names during fit
        self._fitted: bool = False # Track if the model has been successfully fitted

        logger.debug(f"BaseClassifier '{self.model_name}' initialized with config: {self.model_config}")

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Train the classifier on the input data.

        Implementations should handle preprocessing specific to the model (like scaling),
        store the feature names used for training in `self.feature_names_in_`,
        train the underlying `self.model`, and set `self._fitted = True`.

        Args:
            X: DataFrame of features (n_samples, n_features).
            y: Series of target class indices (n_samples,).
            X_val: Optional validation feature DataFrame for early stopping or evaluation during training.
            y_val: Optional validation target Series.

        Returns:
            self: The fitted classifier instance.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate class predictions for new data.

        Implementations must ensure the input DataFrame `X` has the same features
        (in name and order) as used during training (`self.feature_names_in_`),
        apply any necessary preprocessing (like scaling using a fitted scaler),
        and return predictions from `self.model`.

        Args:
            X: DataFrame of features (n_samples, n_features) matching training features.

        Returns:
            NumPy array of predicted class indices (n_samples,).

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If input features mismatch training features.
        """
        if not self._fitted:
            raise RuntimeError(f"Model '{self.model_name}' has not been fitted yet. Call fit() first.")
        self._check_input_features(X) # Validate features before prediction
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate class probability predictions for new data.

        Similar requirements as `predict` regarding feature consistency and preprocessing.

        Args:
            X: DataFrame of features (n_samples, n_features) matching training features.

        Returns:
            NumPy array of shape (n_samples, n_classes) with class probabilities.
            If probabilities are not supported, should raise NotImplementedError or return default values.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If input features mismatch training features.
            NotImplementedError: If the underlying model doesn't support probabilities.
        """
        if not self._fitted:
            raise RuntimeError(f"Model '{self.model_name}' has not been fitted yet. Call fit() first.")
        self._check_input_features(X) # Validate features before prediction
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save the trained model state to disk.

        Implementations should save the `self.model` instance, `self.feature_names_in_`,
        any necessary preprocessing objects (like scalers), and potentially relevant
        configuration details (`self.model_config` or `self.config`).

        Args:
            path: The file path where the model should be saved. The directory
                  will be created if it doesn't exist.

        Raises:
            RuntimeError: If the model has not been fitted.
            IOError: If saving fails.
        """
        if not self._fitted:
            raise RuntimeError(f"Cannot save model '{self.model_name}' because it has not been fitted.")
        try:
             os.makedirs(os.path.dirname(path), exist_ok=True)
             logger.info(f"Saving model '{self.model_name}' to {path}...")
        except OSError as e:
             logger.error(f"Failed to create directory for saving model at {path}: {e}")
             raise IOError(f"Could not create directory for model file: {path}") from e
        pass # Implement actual saving logic in subclasses

    @classmethod
    @abstractmethod
    def load(cls, path: str, config: Optional[Dict[str, Any]] = None) -> 'BaseClassifier':
        """
        Load a trained model state from disk.

        Implementations should load all necessary components saved by the `save` method
        (model, feature names, scalers, config) and return a fully functional,
        fitted instance of the classifier class.

        Args:
            path: Path to the saved model file.
            config: Optional current runtime config, may be used if saved config is incomplete
                    or for context during loading.

        Returns:
            A loaded, fitted instance of the classifier subclass.

        Raises:
            FileNotFoundError: If the model file doesn't exist.
            IOError: If loading fails.
            ValueError: If the loaded state is incompatible or corrupt.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        logger.info(f"Loading model '{cls.__name__}' from {path}...")
        pass # Implement actual loading logic in subclasses


    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance values if the underlying model supports it.

        Returns:
            A dictionary mapping feature names (from `self.feature_names_in_`)
            to their importance scores (float). Returns None if the model doesn't
            support feature importance or hasn't been fitted.
        """
        if not self._fitted:
            logger.warning(f"Cannot get feature importance for '{self.model_name}': Model not fitted.")
            return None
        # Implementation depends on the specific model (e.g., rf.feature_importances_)
        # Subclasses should override this method.
        logger.warning(f"Feature importance not implemented for model type '{self.__class__.__name__}'.")
        return None

    def hyperparameter_optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization (HPO).

        This method should use the provided training and validation data to search
        for the best hyperparameters according to the HPO configuration specified
        in `self.model_config`.

        After finding the best parameters, it should ideally update `self.model_config`
        or the internal state of the classifier instance so that a subsequent call
        to `fit` uses these optimal parameters for the final training run.

        Args:
            X_train: Training feature DataFrame.
            y_train: Training target Series.
            X_val: Validation feature DataFrame.
            y_val: Validation target Series.

        Returns:
            A dictionary containing the best hyperparameters found during the search.

        Raises:
            NotImplementedError: If HPO is not configured or supported for this model.
            ValueError: If HPO configuration is invalid.
        """
        hpo_enabled = self.model_config.get('randomized_search', {}).get('enabled', False) or \
                      self.model_config.get('hyperparameter_optimization', {}).get('enabled', False)

        if not hpo_enabled:
            raise NotImplementedError(f"Hyperparameter optimization is not enabled in the configuration for model '{self.model_name}'.")

        # Subclasses must implement the specific HPO logic (e.g., using RandomizedSearchCV, Optuna).
        raise NotImplementedError(f"Hyperparameter optimization logic not implemented in subclass '{self.__class__.__name__}'.")


    def _check_input_features(self, X: pd.DataFrame):
        """Internal helper to validate input features against trained features."""
        if self.feature_names_in_ is None:
             # Should not happen if model is fitted, but safeguard
             raise RuntimeError(f"Model '{self.model_name}' is fitted but feature names are missing.")

        input_features = X.columns.tolist()
        if len(input_features) != len(self.feature_names_in_):
            raise ValueError(f"Input feature count mismatch for '{self.model_name}'. "
                             f"Expected {len(self.feature_names_in_)} features, got {len(input_features)}. "
                             f"Expected: {self.feature_names_in_}. Got: {input_features}")

        if input_features != self.feature_names_in_:
            # Check if sets are the same but order differs (more flexible)
            if set(input_features) == set(self.feature_names_in_):
                 logger.warning(f"Input features for '{self.model_name}' are in a different order than training. Reordering input.")
                 # Reorder DataFrame columns - ensure this doesn't cause issues downstream
                 try:
                     X_reordered = X[self.feature_names_in_]
                     # This modifies the DataFrame passed in if it's not a copy - careful!
                     # It might be safer to return the reordered df or work on a copy.
                     # For now, assume the caller handles the input DataFrame appropriately.
                     # If prediction methods work on copies or numpy arrays derived after this check, it's fine.
                     if not X.columns.equals(X_reordered.columns): # Check if reordering actually happened
                           # This might be too intrusive, commenting out direct modification
                           # X = X_reordered
                           pass # Let the calling method use X[self.feature_names_in_]

                 except Exception as e:
                     logger.error(f"Failed to reorder input features: {e}")
                     raise ValueError(f"Could not reorder input features for '{self.model_name}'.") from e
            else:
                 missing = set(self.feature_names_in_) - set(input_features)
                 extra = set(input_features) - set(self.feature_names_in_)
                 err_msg = f"Input feature mismatch for '{self.model_name}'."
                 if missing: err_msg += f" Missing: {missing}."
                 if extra: err_msg += f" Unexpected: {extra}."
                 raise ValueError(err_msg)
