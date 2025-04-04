# --- File: drwiggle/models/xgboost.py ---
import logging
import os
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb # Import XGBoost
# Import the early stopping callback
from xgboost.callback import EarlyStopping
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight # Use sample_weight for XGBoost
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder # Needed if labels aren't 0-indexed

from .base import BaseClassifier
from drwiggle.utils.helpers import progress_bar, save_object, load_object # Use helpers for save/load

logger = logging.getLogger(__name__)

class XGBoostClassifier(BaseClassifier):
    """
    XGBoost Classifier for protein flexibility, integrating with BaseClassifier.
    Handles hyperparameter optimization via RandomizedSearchCV if configured.
    Uses callbacks for early stopping (compatible with XGBoost >= 1.6).
    """

    def __init__(self, config: Dict[str, Any], model_name: str):
        super().__init__(config, model_name)
        # Ensure num_classes is set for XGBoost objective
        self.num_classes = self.config.get('binning', {}).get('num_classes', 5)
        if self.num_classes <= 0:
            raise ValueError("Number of classes must be positive for XGBoost multi-class classification.")
        logger.debug(f"XGBoostClassifier initialized for {self.num_classes} classes.")

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Train the XGBoost classifier. Handles HPO if enabled and uses early stopping via callbacks.
        """
        self.feature_names_in_ = X.columns.tolist() # Store feature names

        # --- Data Preparation ---
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y.astype(int))
        if not np.array_equal(le.classes_, np.arange(self.num_classes)):
             logger.warning(f"Labels were re-encoded by LabelEncoder. Original unique labels: {np.unique(y.values)}, Encoded classes: {le.classes_}. Ensure labels match 0 to num_classes-1.")
        self.label_encoder_ = le

        # Prepare validation set and early stopping parameters if applicable
        eval_sets_for_fit = None
        callbacks_for_fit = [] # Use a list for callbacks
        verbose_fit_arg = False # Controls XGBoost's print frequency during training
        early_stopping_rounds_val = self.model_config.get('training', {}).get('early_stopping_rounds', None)
        use_early_stopping = early_stopping_rounds_val and early_stopping_rounds_val > 0

        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder_.transform(y_val.astype(int))
            # XGBoost eval_set needs to be a list of tuples: [(X_train, y_train), (X_val, y_val)]
            # Add training set to eval_sets for monitoring train metric if desired
            eval_sets_for_fit = [
                (X.values, y_train_encoded), # Train set
                (X_val[self.feature_names_in_].values, y_val_encoded) # Validation set
            ]
            logger.info(f"Validation set prepared for XGBoost evaluation. Shape: {X_val.shape}")

            # Configure early stopping callback if enabled
            if use_early_stopping:
                 # verbose parameter in EarlyStopping callback controls messages from the callback itself
                 early_stop_verbose = self.model_config.get('training', {}).get('verbose', False)
                 # Set save_best=True to retain the best model weights based on validation score
                 early_stopping_callback = EarlyStopping(
                     rounds=early_stopping_rounds_val,
                     metric_name=self.model_config.get('eval_metric', 'mlogloss'), # Monitor this metric
                     save_best=True, # Important: store the best model internally
                     maximize=False, # Default: minimize loss metrics (set True for accuracy etc.)
                     # data_name='validation_0' # Name must match eval_sets list index or explicit name
                 )
                 callbacks_for_fit.append(early_stopping_callback)
                 logger.info(f"Using EarlyStopping callback with rounds={early_stopping_rounds_val}, metric='{early_stopping_callback.metric_name}'")
                 # Set XGBoost's fit verbosity based on config if early stopping active
                 verbose_fit_arg = self.model_config.get('training', {}).get('verbose', False)
            else:
                 logger.info("Early stopping rounds not configured or set to zero. Early stopping callback disabled.")
        else:
            logger.warning("No validation set provided for XGBoost. Early stopping disabled.")
            use_early_stopping = False # Explicitly disable if no val set

        # Handle class weights using sample_weight
        sample_weights = None
        use_weights = self.model_config.get('use_sample_weights', True)
        if use_weights:
            logger.debug("Calculating sample weights for imbalanced classes.")
            try:
                sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_encoded)
            except Exception as e:
                 logger.warning(f"Could not compute sample weights: {e}. Proceeding without weights.")

        # --- HPO Phase ---
        hpo_cfg = self.model_config.get('randomized_search', {})
        hpo_enabled = hpo_cfg.get('enabled', False)

        if hpo_enabled:
            logger.info(f"Starting hyperparameter optimization for {self.model_name} using RandomizedSearchCV...")
            try:
                best_params_found = self.hyperparameter_optimize(X, y, X_val, y_val)
                logger.info(f"Updating model config with best HPO params: {best_params_found}")
                self.model_config.update(best_params_found) # Update top-level params
            except (NotImplementedError, ValueError) as e:
                logger.error(f"Hyperparameter optimization failed: {e}. Falling back to default parameters.")

        # --- Final Model Training ---
        logger.info(f"Fitting final {self.model_name} model...")

        # Extract parameters from config (potentially updated by HPO)
        xgb_params = {
            'objective': self.model_config.get('objective', 'multi:softprob'),
            'eval_metric': self.model_config.get('eval_metric', 'mlogloss'), # This is used by eval_set
            'n_estimators': self.model_config.get('n_estimators', 100),
            'learning_rate': self.model_config.get('learning_rate', 0.1),
            'max_depth': self.model_config.get('max_depth', 6),
            'subsample': self.model_config.get('subsample', 0.8),
            'colsample_bytree': self.model_config.get('colsample_bytree', 0.8),
            'gamma': self.model_config.get('gamma', 0),
            'reg_alpha': self.model_config.get('reg_alpha', 0),
            'reg_lambda': self.model_config.get('reg_lambda', 1),
            'use_label_encoder': False, # Deprecated, set False
            'seed': self.config.get('system', {}).get('random_state', 42),
            'n_jobs': self.config.get('system', {}).get('n_jobs', -1),
            # num_class is automatically inferred from data or objective in recent XGBoost
            # 'num_class': self.num_classes # Usually not needed if objective is multi:*
        }
        # Add num_class explicitly if required by older XGBoost versions or specific objectives
        if xgb_params['objective'].startswith('multi:'):
             xgb_params['num_class'] = self.num_classes

        logger.debug(f"Final XGBoost training parameters: {xgb_params}")

        self.model = xgb.XGBClassifier(**xgb_params)

        try:
            # Call fit, passing eval_set and callbacks if they exist
            self.model.fit(
                X.values,
                y_train_encoded,
                sample_weight=sample_weights,
                eval_set=eval_sets_for_fit, # Pass the list of validation sets
                callbacks=callbacks_for_fit if callbacks_for_fit else None, # Pass the list of callbacks
                verbose=verbose_fit_arg # Controls XGBoost's own print frequency
            )

            logger.info(f"Final {self.model_name} training complete.")
            # Log best score if early stopping was used and the attribute exists
            if use_early_stopping and hasattr(self.model, 'best_score'):
                 try:
                      best_score_val = self.model.best_score
                      best_iter_val = self.model.best_iteration
                      eval_metric_name = xgb_params['eval_metric'] # Use the metric from params
                      # If multiple metrics were tracked, best_score might be the last one. Check results dict?
                      # results = self.model.evals_result() # Contains metrics history
                      logger.info(f"Best validation score ({eval_metric_name}) during early stopping: {best_score_val:.4f} at iteration {best_iter_val}")
                 except AttributeError:
                      logger.info("Early stopping used, but best_score/best_iteration attribute not found (might depend on XGBoost version or exact callback setup).")

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
            # Use best_iteration if early stopping saved the best model
            iteration_range = (0, self.model.best_iteration + 1) if hasattr(self.model, 'best_iteration') and self.model.best_iteration else (0, 0)

            encoded_predictions = self.model.predict(X[self.feature_names_in_].values, iteration_range=iteration_range)
            return encoded_predictions
        except NotFittedError:
             raise RuntimeError(f"Internal error: Model '{self.model_name}' predict called but underlying XGBoost model is not fitted.")
        except Exception as e:
            logger.error(f"Error during {self.model_name} prediction: {e}", exc_info=True)
            raise

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate class probabilities."""
        super().predict_proba(X) # Check if fitted and features match
        try:
            if not hasattr(self.model, 'predict_proba'):
                raise NotImplementedError(f"The underlying {self.model.__class__.__name__} model does not support probability prediction (check objective).")

            # Use best_iteration if early stopping saved the best model
            iteration_range = (0, self.model.best_iteration + 1) if hasattr(self.model, 'best_iteration') and self.model.best_iteration else (0, 0)

            probabilities = self.model.predict_proba(X[self.feature_names_in_].values, iteration_range=iteration_range)
            return probabilities
        except NotFittedError:
             raise RuntimeError(f"Internal error: Model '{self.model_name}' predict_proba called but underlying XGBoost model is not fitted.")
        except Exception as e:
            logger.error(f"Error during {self.model_name} probability prediction: {e}", exc_info=True)
            raise

    def save(self, path: str):
        """Save the trained model state using joblib via helper."""
        super().save(path) # Creates dir, checks fitted state
        if not hasattr(self, 'label_encoder_'):
            logger.warning(f"Label encoder not found for model '{self.model_name}'. Saving without it.")
            label_encoder_state = None
        else:
            label_encoder_state = self.label_encoder_

        state = {
            'model': self.model, # XGBoost model object itself
            'label_encoder_': label_encoder_state,
            'feature_names_in_': self.feature_names_in_,
            'config': self.config,
            'model_config': self.model_config,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'fitted': self._fitted
        }
        save_object(state, path) # Use helper

    @classmethod
    def load(cls, path: str, config: Optional[Dict[str, Any]] = None) -> 'XGBoostClassifier':
        """Load a trained model state using joblib via helper."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        logger.info(f"Loading model '{cls.__name__}' from {path}...")
        state = load_object(path) # Use helper

        required_keys = ['model', 'feature_names_in_', 'model_config', 'model_name', 'fitted', 'num_classes']
        if not all(key in state for key in required_keys):
             raise ValueError(f"Loaded model state from {path} is missing required keys. Found: {list(state.keys())}")

        load_config = state.get('config', config)
        if load_config is None:
             raise ValueError("Cannot load model: No configuration found in saved state or provided at runtime.")

        instance = cls(config=load_config, model_name=state['model_name'])

        # Restore state
        instance.model = state['model']
        instance.label_encoder_ = state.get('label_encoder_')
        instance.feature_names_in_ = state['feature_names_in_']
        instance.num_classes = state['num_classes']
        instance._fitted = state['fitted']
        # instance.model_config = state['model_config'] # Optionally restore exact config

        if not instance._fitted:
             logger.warning(f"Loaded model '{instance.model_name}' from {path} indicates it was not fitted.")
        if not isinstance(instance.model, xgb.XGBClassifier):
             raise TypeError(f"Loaded model is not an XGBoostClassifier instance (got {type(instance.model)}).")

        logger.info(f"Model '{instance.model_name}' loaded successfully.")
        return instance

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance values from the trained XGBoost model."""
        if not self._fitted or not hasattr(self.model, 'feature_importances_'):
            logger.warning(f"Cannot get feature importance for '{self.model_name}': Model not fitted or importances not available.")
            return None

        importances = self.model.feature_importances_

        if self.feature_names_in_ and len(self.feature_names_in_) == len(importances):
            importance_dict = dict(zip(self.feature_names_in_, importances))
            sorted_importances = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
            return sorted_importances
        else:
            logger.warning(f"Feature names mismatch importance values for '{self.model_name}'. Returning indexed importances.")
            return {f"feature_{i}": imp for i, imp in enumerate(importances)}

    def hyperparameter_optimize(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """
        Performs Hyperparameter Optimization using RandomizedSearchCV for XGBoost.
        """
        hpo_cfg = self.model_config.get('randomized_search')
        if not hpo_cfg or not hpo_cfg.get('enabled', False):
             raise NotImplementedError(f"RandomizedSearch HPO not enabled or configured for model '{self.model_name}'.")

        param_dist = hpo_cfg.get('param_distributions')
        n_iter = hpo_cfg.get('n_iter', 20)
        cv = hpo_cfg.get('cv', 3)
        scoring = hpo_cfg.get('scoring', 'balanced_accuracy') # Appropriate scoring

        if not param_dist:
            raise ValueError("Parameter distributions ('param_distributions') not defined in HPO config.")

        logger.info(f"Running RandomizedSearchCV for {self.model_name}: n_iter={n_iter}, cv={cv}, scoring='{scoring}'")
        logger.debug(f"Search space: {param_dist}")

        # Base XGBoost estimator for search - use minimal settings here
        base_estimator = xgb.XGBClassifier(
             objective='multi:softprob', # Consistent objective
             eval_metric='mlogloss',     # Consistent metric
             use_label_encoder=False,    # Deprecated, set False
             seed=self.config.get('system', {}).get('random_state', 42),
             n_jobs=1, # Let RandomizedSearchCV handle parallelization
             # num_class will be inferred or can be set if needed
        )
        # Explicitly set num_class if objective requires it
        if base_estimator.objective.startswith('multi:'):
             base_estimator.set_params(num_class=self.num_classes)


        # Encode labels for RandomizedSearchCV internal fitting
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train.astype(int))

        # Calculate sample weights for CV folds if desired
        fit_params_cv = {} # Parameters passed to fit within CV
        use_weights_hpo = self.model_config.get('use_sample_weights', True)
        if use_weights_hpo:
             sample_weights_hpo = compute_sample_weight(class_weight='balanced', y=y_train_encoded)
             fit_params_cv['sample_weight'] = sample_weights_hpo
             logger.info("Using sample weights during HPO cross-validation.")

        # NOTE: RandomizedSearchCV doesn't easily support early stopping with external validation set.
        # It uses cross-validation internally. If you need early stopping during HPO,
        # you might consider Optuna or a custom HPO loop. Here we rely on CV performance.

        search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=self.config.get('system', {}).get('random_state', 42),
            n_jobs=self.config.get('system', {}).get('n_jobs', -1),
            verbose=1,
            error_score='raise' # Catch errors during search
        )

        try:
            # Run the search on training data, passing fit_params to the estimator within CV
            search.fit(X_train.values, y_train_encoded, **fit_params_cv)

            logger.info(f"RandomizedSearch complete. Best score ({scoring}): {search.best_score_:.4f}")
            logger.info(f"Best parameters found: {search.best_params_}")

            # Return only the best hyperparameters
            return search.best_params_

        except Exception as e:
            logger.error(f"RandomizedSearchCV failed for {self.model_name}: {e}", exc_info=True)
            raise ValueError(f"Hyperparameter optimization failed during RandomizedSearch.") from e