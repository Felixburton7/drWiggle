import logging
import os
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.exceptions import NotFittedError

from .base import BaseClassifier
from drwiggle.utils.helpers import progress_bar, save_object, load_object # Use helpers for save/load

logger = logging.getLogger(__name__)

class RandomForestClassifier(BaseClassifier):
    """
    Random Forest Classifier for protein flexibility, integrating with BaseClassifier.
    Handles hyperparameter optimization via RandomizedSearchCV if configured.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Train the Random Forest classifier. Handles HPO if enabled.

        Args:
            X: DataFrame of features.
            y: Series of target labels.
            X_val: Validation features (currently unused by RF fit/HPO but kept for interface consistency).
            y_val: Validation labels (currently unused).
        """
        self.feature_names_in_ = X.columns.tolist() # Store feature names

        # Determine if HPO is enabled for RF
        hpo_cfg = self.model_config.get('randomized_search', {})
        hpo_enabled = hpo_cfg.get('enabled', False)

        best_params_found = None

        if hpo_enabled:
            logger.info(f"Starting hyperparameter optimization for {self.model_name} using RandomizedSearchCV...")
            try:
                 # Run HPO - this method now returns the best params but doesn't fit the final model yet
                 best_params_found = self.hyperparameter_optimize(X, y, X_val, y_val)
                 # Update model_config with best parameters found for final fit
                 logger.info(f"Updating model config with best HPO params: {best_params_found}")
                 for key, value in best_params_found.items():
                     # Need to place params correctly in config structure (e.g., n_estimators directly)
                      if key in self.model_config:
                          self.model_config[key] = value
                      else:
                           # Handle nested params if necessary, though RF params are usually top-level
                           logger.warning(f"Best HPO param '{key}' not directly in model_config, check structure.")

            except (NotImplementedError, ValueError) as e:
                 logger.error(f"Hyperparameter optimization failed: {e}. Falling back to default parameters.")
                 hpo_enabled = False # Disable HPO for the final fit stage


        # --- Final Model Training ---
        # Use either default params or best params found from HPO
        logger.info(f"Fitting final {self.model_name} model...")
        final_params = {
            'n_estimators': self.model_config.get('n_estimators', 100),
            'max_depth': self.model_config.get('max_depth', None),
            'min_samples_split': self.model_config.get('min_samples_split', 2),
            'min_samples_leaf': self.model_config.get('min_samples_leaf', 1),
            'class_weight': self.model_config.get('class_weight', 'balanced'),
            'random_state': self.config.get('system', {}).get('random_state', 42),
            'n_jobs': self.config.get('system', {}).get('n_jobs', -1),
            'oob_score': True # Useful for assessing performance without a separate val set during fit
        }
        logger.debug(f"Final RF training parameters: {final_params}")

        self.model = SklearnRF(**final_params)

        try:
            # Ensure y has the correct integer type expected by sklearn
            y_train_values = y.astype(int).values
            self.model.fit(X.values, y_train_values) # Sklearn models typically work best with numpy arrays
            logger.info(f"Final {self.model_name} training complete. OOB Score: {self.model.oob_score_:.4f}")
            self._fitted = True
        except Exception as e:
            logger.error(f"Failed to train final {self.model_name} model: {e}", exc_info=True)
            self._fitted = False
            raise # Re-raise the exception

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate class predictions."""
        super().predict(X) # Check if fitted and features match
        try:
            # Use .values to ensure numpy array input for sklearn model
            predictions = self.model.predict(X[self.feature_names_in_].values)
            return predictions
        except NotFittedError:
             raise RuntimeError(f"Internal error: Model '{self.model_name}' predict called but underlying sklearn model is not fitted.")
        except Exception as e:
            logger.error(f"Error during {self.model_name} prediction: {e}", exc_info=True)
            raise

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate class probabilities."""
        super().predict_proba(X) # Check if fitted and features match
        try:
            if not hasattr(self.model, 'predict_proba'):
                raise NotImplementedError(f"The underlying {self.model.__class__.__name__} model does not support probability prediction.")
            # Use .values to ensure numpy array input
            probabilities = self.model.predict_proba(X[self.feature_names_in_].values)
            return probabilities
        except NotFittedError:
             raise RuntimeError(f"Internal error: Model '{self.model_name}' predict_proba called but underlying sklearn model is not fitted.")
        except Exception as e:
            logger.error(f"Error during {self.model_name} probability prediction: {e}", exc_info=True)
            raise

    def save(self, path: str):
        """Save the trained model state using joblib via helper."""
        super().save(path) # Creates dir, checks fitted state
        state = {
            'model': self.model,
            'feature_names_in_': self.feature_names_in_,
            'config': self.config, # Save full config used during training instance creation
            'model_config': self.model_config, # Save specific model config state at time of saving
            'model_name': self.model_name,
            'fitted': self._fitted
        }
        save_object(state, path) # Use helper

    @classmethod
    def load(cls, path: str, config: Optional[Dict[str, Any]] = None) -> 'RandomForestClassifier':
        """Load a trained model state using joblib via helper."""
        # Base class load method only checks file existence in this implementation
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        logger.info(f"Loading model '{cls.__name__}' from {path}...")
        state = load_object(path) # Use helper

        # Validate loaded state (basic checks)
        required_keys = ['model', 'feature_names_in_', 'model_config', 'model_name', 'fitted']
        if not all(key in state for key in required_keys):
             raise ValueError(f"Loaded model state from {path} is missing required keys. Found: {list(state.keys())}")

        # Use loaded config if available, otherwise fall back to provided runtime config
        load_config = state.get('config', config)
        if load_config is None:
             raise ValueError("Cannot load model: No configuration found in saved state or provided at runtime.")

        # Re-instantiate the class
        instance = cls(config=load_config, model_name=state['model_name'])

        # Restore state
        instance.model = state['model']
        instance.feature_names_in_ = state['feature_names_in_']
        instance._fitted = state['fitted']
        # Optionally restore model_config from state if needed, but typically re-derived from main config
        # instance.model_config = state['model_config']

        if not instance._fitted:
             logger.warning(f"Loaded model '{instance.model_name}' from {path} indicates it was not fitted.")
        if not isinstance(instance.model, SklearnRF):
             raise TypeError(f"Loaded model is not a scikit-learn RandomForestClassifier instance (got {type(instance.model)}).")

        logger.info(f"Model '{instance.model_name}' loaded successfully.")
        return instance

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance values from the trained RF model."""
        if not self._fitted or not hasattr(self.model, 'feature_importances_'):
            logger.warning(f"Cannot get feature importance for '{self.model_name}': Model not fitted or importances not available.")
            return None

        importances = self.model.feature_importances_

        if self.feature_names_in_ and len(self.feature_names_in_) == len(importances):
            # Create dict mapping names to scores
            importance_dict = dict(zip(self.feature_names_in_, importances))
            # Sort by importance (descending)
            sorted_importances = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
            return sorted_importances
        else:
            logger.warning(f"Feature names ({len(self.feature_names_in_) if self.feature_names_in_ else 'None'}) "
                           f"mismatch importance values ({len(importances)}) for model '{self.model_name}'. Returning indexed importances.")
            # Fallback to indexed importances if names mismatch
            return {f"feature_{i}": imp for i, imp in enumerate(importances)}

    def hyperparameter_optimize(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """
        Performs Hyperparameter Optimization using RandomizedSearchCV.

        Note: Validation data (X_val, y_val) is not directly used by RandomizedSearchCV's
        internal cross-validation, but the method signature requires them for consistency
        with the base class.

        Returns:
            Dictionary containing the best hyperparameters found.
        """
        hpo_cfg = self.model_config.get('randomized_search')
        if not hpo_cfg or not hpo_cfg.get('enabled', False):
             raise NotImplementedError(f"RandomizedSearch HPO not enabled or configured for model '{self.model_name}'.")

        param_dist = hpo_cfg.get('param_distributions')
        n_iter = hpo_cfg.get('n_iter', 20)
        cv = hpo_cfg.get('cv', 3)
        scoring = hpo_cfg.get('scoring', 'balanced_accuracy') # Default scoring metric

        if not param_dist:
            raise ValueError("Parameter distributions ('param_distributions') not defined in HPO config.")

        logger.info(f"Running RandomizedSearchCV for {self.model_name}: n_iter={n_iter}, cv={cv}, scoring='{scoring}'")
        logger.debug(f"Search space: {param_dist}")

        # Base estimator (parameters here are defaults, will be overridden by search)
        base_estimator = SklearnRF(
             random_state=self.config.get('system', {}).get('random_state', 42),
             n_jobs=1 # Use n_jobs in RandomizedSearchCV for parallelism across CV folds/iters
        )

        # Ensure y has the correct integer type
        y_train_values = y_train.astype(int).values

        search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=self.config.get('system', {}).get('random_state', 42),
            n_jobs=self.config.get('system', {}).get('n_jobs', -1), # Parallelize search
            verbose=1 # Show some progress from RandomizedSearchCV
        )

        try:
            # Run the search on training data (uses internal CV)
            # Need .values as sklearn works best with numpy
            search.fit(X_train.values, y_train_values)

            logger.info(f"RandomizedSearch complete. Best score ({scoring}): {search.best_score_:.4f}")
            logger.info(f"Best parameters found: {search.best_params_}")

            # This method *only* finds the best params, it doesn't fit the final model here.
            # The calling `fit` method will use these params to train the final model.
            return search.best_params_

        except Exception as e:
            logger.error(f"RandomizedSearchCV failed for {self.model_name}: {e}", exc_info=True)
            raise ValueError(f"Hyperparameter optimization failed during RandomizedSearch.") from e
