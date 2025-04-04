# --- File: drwiggle/models/lightgbm_model.py ---
import logging
import os
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb # Import LightGBM
from sklearn.model_selection import RandomizedSearchCV
# LightGBM handles balanced classes internally, so sample_weight might not be needed
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder

from .base import BaseClassifier
from drwiggle.utils.helpers import progress_bar, save_object, load_object

logger = logging.getLogger(__name__)

class LightGBMClassifier(BaseClassifier):
    """
    LightGBM Classifier for protein flexibility, integrating with BaseClassifier.
    Handles hyperparameter optimization via RandomizedSearchCV if configured.
    """

    def __init__(self, config: Dict[str, Any], model_name: str):
        super().__init__(config, model_name)
        self.num_classes = self.config.get('binning', {}).get('num_classes', 5)
        if self.num_classes <= 0:
            raise ValueError("Number of classes must be positive for LightGBM multi-class classification.")
        # LightGBM uses 'num_class' parameter
        logger.debug(f"LightGBMClassifier initialized for {self.num_classes} classes.")

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Train the LightGBM classifier. Handles HPO if enabled and uses early stopping.

        Args:
            X: DataFrame of features.
            y: Series of target labels (must be 0-indexed integers).
            X_val: Validation features for early stopping.
            y_val: Validation labels for early stopping.
        """
        self.feature_names_in_ = X.columns.tolist()

        # --- Data Preparation ---
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y.astype(int))
        if not np.array_equal(le.classes_, np.arange(self.num_classes)):
             logger.warning(f"Labels were re-encoded by LabelEncoder. Original unique labels: {np.unique(y.values)}, Encoded classes: {le.classes_}. Ensure labels match 0 to num_classes-1.")
        self.label_encoder_ = le

        eval_set = None
        eval_metric = self.model_config.get('eval_metric', 'multi_logloss') # Default metric
        fit_params = {} # Parameters passed directly to lgb.fit()

        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder_.transform(y_val.astype(int))
            # LightGBM needs eval_set as a list of tuples
            eval_set = [(X_val[self.feature_names_in_].values, y_val_encoded)]
            fit_params['eval_set'] = eval_set
            fit_params['eval_names'] = ['validation'] # Name for the eval set
            fit_params['eval_metric'] = eval_metric # Metric for early stopping
            logger.info(f"Validation set prepared for LightGBM early stopping. Shape: {X_val.shape}")

            early_stopping_rounds = self.model_config.get('training', {}).get('early_stopping_rounds', None)
            if early_stopping_rounds:
                 # Use callbacks for early stopping in LightGBM >= 4.0
                 # For older versions (<4.0), early_stopping_rounds was a direct fit parameter
                 try:
                     # Try using the callback API (LGBM >= 4.0)
                     fit_params['callbacks'] = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=self.model_config.get('training', {}).get('verbose', False))]
                     logger.info(f"Using LightGBM callback API for early stopping with rounds={early_stopping_rounds}")
                 except AttributeError:
                      # Fallback for older LightGBM versions
                      fit_params['early_stopping_rounds'] = early_stopping_rounds
                      fit_params['verbose'] = self.model_config.get('training', {}).get('verbose', False)
                      logger.info(f"Using LightGBM fit parameter for early stopping with rounds={early_stopping_rounds} (older LGBM version detected).")


        else:
            logger.warning("No validation set provided for LightGBM. Early stopping disabled.")


        # --- HPO Phase ---
        hpo_cfg = self.model_config.get('randomized_search', {})
        hpo_enabled = hpo_cfg.get('enabled', False)

        if hpo_enabled:
            logger.info(f"Starting hyperparameter optimization for {self.model_name} using RandomizedSearchCV...")
            try:
                best_params_found = self.hyperparameter_optimize(X, y, X_val, y_val)
                logger.info(f"Updating model config with best HPO params: {best_params_found}")
                self.model_config.update(best_params_found)
            except (NotImplementedError, ValueError) as e:
                logger.error(f"Hyperparameter optimization failed: {e}. Falling back to default parameters.")


        # --- Final Model Training ---
        logger.info(f"Fitting final {self.model_name} model...")

        # Extract parameters from config (potentially updated by HPO)
        lgb_params = {
            'objective': self.model_config.get('objective', 'multiclass'),
            'metric': eval_metric, # Use the same metric for internal tracking
            'num_class': self.num_classes,
            'n_estimators': self.model_config.get('n_estimators', 100),
            'learning_rate': self.model_config.get('learning_rate', 0.1),
            'num_leaves': self.model_config.get('num_leaves', 31), # Important LGBM param
            'max_depth': self.model_config.get('max_depth', -1), # -1 means no limit (use num_leaves)
            'subsample': self.model_config.get('subsample', 0.8), # Aliases: bagging_fraction
            'colsample_bytree': self.model_config.get('colsample_bytree', 0.8), # Aliases: feature_fraction
            'reg_alpha': self.model_config.get('reg_alpha', 0), # L1
            'reg_lambda': self.model_config.get('reg_lambda', 0), # L2
            # Class weight handling
            'class_weight': 'balanced' if self.model_config.get('class_weight', 'balanced') == 'balanced' else None,
            # 'is_unbalance': True if self.model_config.get('is_unbalance', False) else False, # Alternative way to handle imbalance
            'random_state': self.config.get('system', {}).get('random_state', 42),
            'n_jobs': self.config.get('system', {}).get('n_jobs', -1),
            'verbose': -1 # Suppress LightGBM default verbosity unless verbose in fit_params is set
        }
        logger.debug(f"Final LightGBM training parameters: {lgb_params}")

        self.model = lgb.LGBMClassifier(**lgb_params)

        try:
            # Fit the model with numpy arrays and optional early stopping params
            self.model.fit(X.values, y_train_encoded, **fit_params)
            logger.info(f"Final {self.model_name} training complete.")
            # Log best score if early stopping was used
            if 'callbacks' in fit_params or 'early_stopping_rounds' in fit_params:
                if hasattr(self.model, 'best_score_') and self.model.best_score_:
                     # LGBM stores best scores per eval set and metric
                     best_score_dict = self.model.best_score_
                     val_metric_key = next(iter(best_score_dict.get('validation', {})), None) # Get the first metric name used
                     if val_metric_key:
                         best_score = best_score_dict['validation'][val_metric_key]
                         logger.info(f"Best score ({val_metric_key}) during early stopping: {best_score:.4f} at iteration {self.model.best_iteration_}")
            self._fitted = True
        except Exception as e:
            logger.error(f"Failed to train final {self.model_name} model: {e}", exc_info=True)
            self._fitted = False
            raise

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate class predictions."""
        super().predict(X)
        try:
            encoded_predictions = self.model.predict(X[self.feature_names_in_].values)
            # Return 0-indexed predictions directly
            return encoded_predictions
        except NotFittedError:
             raise RuntimeError(f"Internal error: Model '{self.model_name}' predict called but underlying LightGBM model is not fitted.")
        except Exception as e:
            logger.error(f"Error during {self.model_name} prediction: {e}", exc_info=True)
            raise

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate class probabilities."""
        super().predict_proba(X)
        try:
            if not hasattr(self.model, 'predict_proba'):
                raise NotImplementedError(f"The underlying {self.model.__class__.__name__} model does not support probability prediction.")
            probabilities = self.model.predict_proba(X[self.feature_names_in_].values)
            return probabilities
        except NotFittedError:
             raise RuntimeError(f"Internal error: Model '{self.model_name}' predict_proba called but underlying LightGBM model is not fitted.")
        except Exception as e:
            logger.error(f"Error during {self.model_name} probability prediction: {e}", exc_info=True)
            raise

    def save(self, path: str):
        """Save the trained model state using joblib."""
        super().save(path)
        if not hasattr(self, 'label_encoder_'):
            logger.warning(f"Label encoder not found for model '{self.model_name}'. Saving without it.")
            label_encoder_state = None
        else:
            label_encoder_state = self.label_encoder_

        state = {
            'model': self.model, # LightGBM model object
            'label_encoder_': label_encoder_state,
            'feature_names_in_': self.feature_names_in_,
            'config': self.config,
            'model_config': self.model_config,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'fitted': self._fitted
        }
        save_object(state, path)

    @classmethod
    def load(cls, path: str, config: Optional[Dict[str, Any]] = None) -> 'LightGBMClassifier':
        """Load a trained model state using joblib."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        logger.info(f"Loading model '{cls.__name__}' from {path}...")
        state = load_object(path)

        required_keys = ['model', 'feature_names_in_', 'model_config', 'model_name', 'fitted', 'num_classes']
        if not all(key in state for key in required_keys):
             raise ValueError(f"Loaded model state from {path} is missing required keys. Found: {list(state.keys())}")

        load_config = state.get('config', config)
        if load_config is None:
             raise ValueError("Cannot load model: No configuration found in saved state or provided at runtime.")

        instance = cls(config=load_config, model_name=state['model_name'])

        instance.model = state['model']
        instance.label_encoder_ = state.get('label_encoder_')
        instance.feature_names_in_ = state['feature_names_in_']
        instance.num_classes = state['num_classes']
        instance._fitted = state['fitted']
        # instance.model_config = state['model_config']

        if not instance._fitted:
             logger.warning(f"Loaded model '{instance.model_name}' from {path} indicates it was not fitted.")
        if not isinstance(instance.model, lgb.LGBMClassifier):
             raise TypeError(f"Loaded model is not a LightGBM LGBMClassifier instance (got {type(instance.model)}).")

        logger.info(f"Model '{instance.model_name}' loaded successfully.")
        return instance

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance values from the trained LightGBM model."""
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
        Performs Hyperparameter Optimization using RandomizedSearchCV for LightGBM.
        """
        hpo_cfg = self.model_config.get('randomized_search')
        if not hpo_cfg or not hpo_cfg.get('enabled', False):
             raise NotImplementedError(f"RandomizedSearch HPO not enabled or configured for model '{self.model_name}'.")

        param_dist = hpo_cfg.get('param_distributions')
        n_iter = hpo_cfg.get('n_iter', 20)
        cv = hpo_cfg.get('cv', 3)
        scoring = hpo_cfg.get('scoring', 'balanced_accuracy')

        if not param_dist:
            raise ValueError("Parameter distributions ('param_distributions') not defined in HPO config.")

        logger.info(f"Running RandomizedSearchCV for {self.model_name}: n_iter={n_iter}, cv={cv}, scoring='{scoring}'")
        logger.debug(f"Search space: {param_dist}")

        # Base LightGBM estimator for search
        base_estimator = lgb.LGBMClassifier(
             objective='multiclass',
             metric='multi_logloss', # Internal metric, scoring param used by CV
             num_class=self.num_classes,
             class_weight='balanced', # Apply balancing within CV fits
             random_state=self.config.get('system', {}).get('random_state', 42),
             n_jobs=1, # Let RandomizedSearchCV handle parallelization
             verbose=-1 # Suppress verbosity during HPO fits
        )

        # Encode labels for CV
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train.astype(int))

        # Note: RandomizedSearchCV doesn't easily support sample_weight per fold AND early stopping based on external val set.
        # Relying on 'class_weight' in the estimator is the simplest approach here for CV.

        search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=self.config.get('system', {}).get('random_state', 42),
            n_jobs=self.config.get('system', {}).get('n_jobs', -1),
            verbose=1,
            error_score='raise'
        )

        try:
            # Run the search on training data
            search.fit(X_train.values, y_train_encoded) # No fit_params needed if using class_weight

            logger.info(f"RandomizedSearch complete. Best score ({scoring}): {search.best_score_:.4f}")
            logger.info(f"Best parameters found: {search.best_params_}")

            # Return only the best hyperparameters
            return search.best_params_

        except Exception as e:
            logger.error(f"RandomizedSearchCV failed for {self.model_name}: {e}", exc_info=True)
            raise ValueError(f"Hyperparameter optimization failed during RandomizedSearch.") from e