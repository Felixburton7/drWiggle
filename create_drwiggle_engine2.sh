#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if we are in the correct directory (drwiggle_project)
if [[ ! -f "setup.py" || ! -d "drwiggle" ]]; then
    echo "Error: This script must be run from the 'drwiggle_project' directory created by Script 1."
    echo "Current directory: $(pwd)"
    exit 1
fi

echo "--- drWiggle Engine Generation: Script 2 ---"
echo "Generating model implementations, utilities, pipeline, and CLI..."

echo "Creating models/random_forest.py..."
cat << 'EOF' > drwiggle/models/random_forest.py
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
EOF

echo "Creating models/neural_network.py..."
cat << 'EOF' > drwiggle/models/neural_network.py
import logging
import os
import time
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd
import joblib
import warnings

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Other necessary imports
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.exceptions import NotFittedError

# Optuna for HPO
try:
    import optuna
    # Check if progress bar integration is available (Optuna v3+)
    _optuna_supports_progbar = hasattr(optuna.study.Study, 'optimize') and \
                               'show_progress_bar' in optuna.study.Study.optimize.__code__.co_varnames
    _optuna_available = True
except ImportError:
    optuna = None
    _optuna_supports_progbar = False
    _optuna_available = False

# Local imports
from .base import BaseClassifier
from drwiggle.utils.helpers import progress_bar, save_object, load_object # Use helpers

logger = logging.getLogger(__name__)

# --- Neural Network Module Definition ---
class DrWiggleNN(nn.Module):
    """Feed-forward Neural Network for drWiggle classification."""
    def __init__(self, input_dim: int, num_classes: int, nn_config: Dict[str, Any]):
        """
        Args:
            input_dim: Number of input features.
            num_classes: Number of output classes.
            nn_config: The specific configuration dictionary for the neural_network model.
        """
        super().__init__()
        self.nn_config = nn_config # Store config for easy access
        self.num_classes = num_classes
        arch_cfg = nn_config.get('architecture', {})
        hidden_layers = arch_cfg.get('hidden_layers', [64, 32])
        dropout_rate = arch_cfg.get('dropout', 0.2)
        activation_str = arch_cfg.get('activation', 'relu').lower()
        self.is_ordinal = arch_cfg.get('ordinal_output', False) # Check config

        # Activation function
        if activation_str == 'relu':
            activation_fn = nn.ReLU()
        elif activation_str == 'leaky_relu':
            activation_fn = nn.LeakyReLU()
        # Add other activations like Tanh, Sigmoid if needed
        # elif activation_str == 'tanh':
        #    activation_fn = nn.Tanh()
        else:
            logger.warning(f"Activation function '{activation_str}' not recognized. Using ReLU.")
            activation_fn = nn.ReLU()

        # Build layers dynamically
        layers = []
        last_dim = input_dim
        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            # Consider LayerNorm or BatchNorm here if needed
            # layers.append(nn.BatchNorm1d(hidden_dim)) # Use BatchNorm for NNs
            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim

        # Create the sequential network
        self.network = nn.Sequential(*layers)

        # Define the final output layer
        if self.is_ordinal:
             # For ordinal regression, a common approach is to output num_classes - 1 logits
             # representing the boundaries between classes. A simpler (less correct) approach
             # used as a placeholder might output a single value to regress against class index.
             # We'll use the single output for simplicity here, requiring MSE loss later.
             # Proper ordinal methods (CORAL, CORN) are more involved.
             self.output_layer = nn.Linear(last_dim, 1) # Predict single value (class index proxy)
             logger.info("Configured NN output layer for Ordinal Regression (predicting single value).")
        else:
            # Standard classification: output logits for each class
            self.output_layer = nn.Linear(last_dim, num_classes)
            logger.info(f"Configured NN output layer for Standard Classification ({num_classes} classes).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.network(x)
        x = self.output_layer(x)
        return x

# --- Main Classifier Class ---
class NeuralNetworkClassifier(BaseClassifier):
    """Neural Network classifier implementing the BaseClassifier interface."""

    def __init__(self, config: Dict[str, Any], model_name: str):
        super().__init__(config, model_name)
        self.device = self._get_device()
        self.scaler = StandardScaler() # Scaler for input features
        self.num_classes = self.config.get('binning', {}).get('num_classes', 5)
        self.history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': [], 'val_accuracy': []} # Track training progress
        self.best_val_metric = float('inf') if self._is_loss_objective() else float('-inf') # Track best validation metric during training
        self.best_model_state_dict: Optional[Dict[str, torch.Tensor]] = None

    def _get_device(self) -> torch.device:
        """Determine compute device (GPU or CPU) based on config and availability."""
        gpu_enabled_cfg = self.config.get("system", {}).get("gpu_enabled", "auto")
        force_cpu = gpu_enabled_cfg == False
        force_gpu = gpu_enabled_cfg == True
        auto_detect = gpu_enabled_cfg == "auto"

        if not force_cpu and torch.cuda.is_available():
             logger.info("CUDA (Nvidia GPU) is available and selected.")
             return torch.device("cuda")
        elif not force_cpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
             # Check MPS availability specifically for Apple Silicon
             is_built = torch.backends.mps.is_built()
             if is_built:
                logger.info("MPS (Apple Silicon GPU) is available and selected.")
                return torch.device("mps")
             else:
                logger.warning("MPS available but not built correctly? Falling back.")

        elif force_gpu:
             # If GPU explicitly requested but none found
             raise RuntimeError("GPU usage requested (system.gpu_enabled=true), but no compatible GPU (CUDA/MPS) found.")

        # Default to CPU if auto and no GPU, or if explicitly forced
        logger.info("Using CPU.")
        return torch.device("cpu")

    def _is_loss_objective(self) -> bool:
        """Check if the HPO/early stopping objective is loss (needs minimization)."""
        # Default HPO objective is val_loss
        hpo_metric = self.model_config.get('hyperparameter_optimization', {}).get('objective_metric', 'val_loss')
        # Early stopping is usually based on val_loss
        return 'loss' in hpo_metric.lower()

    def _get_optimizer(self, model_params) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_name = self.model_config.get('training', {}).get('optimizer', 'adam').lower()
        lr = float(self.model_config.get('training', {}).get('learning_rate', 0.001)) # Ensure float

        if optimizer_name == 'adam':
            return optim.Adam(model_params, lr=lr)
        elif optimizer_name == 'adamw': # Add AdamW as good alternative
             return optim.AdamW(model_params, lr=lr)
        elif optimizer_name == 'sgd':
             # Add momentum commonly used with SGD
             momentum = float(self.model_config.get('training', {}).get('sgd_momentum', 0.9))
             return optim.SGD(model_params, lr=lr, momentum=momentum)
        else:
            logger.warning(f"Optimizer '{optimizer_name}' not recognized. Using Adam.")
            return optim.Adam(model_params, lr=lr)

    def _get_loss_function(self, y_train: Optional[np.ndarray] = None) -> nn.Module:
        """
        Create loss function. Handles class weights for classification and
        uses MSE as a *placeholder* for ordinal regression.
        """
        arch_cfg = self.model_config.get('architecture', {})
        is_ordinal = arch_cfg.get('ordinal_output', False)

        if is_ordinal:
            # Using MSE on predicted value vs target class index.
            # !!! THIS IS NOT CORRECT ORDINAL REGRESSION but a simple starting point. !!!
            # Replace with proper ordinal loss like CORN or CORAL if implementing seriously.
            logger.warning("Using MSE loss as a placeholder for ordinal regression. This is not theoretically sound. Implement CORN/CORAL loss for proper ordinal handling.")
            return nn.MSELoss()
        else:
            # Standard CrossEntropyLoss for classification
            use_class_weights = self.model_config.get('training', {}).get('class_weights', True)
            weights = None
            if use_class_weights and y_train is not None and len(y_train) > 0:
                try:
                    unique_classes = np.unique(y_train)
                    if len(unique_classes) > 1: # Avoid error if only one class present
                        weights_np = compute_class_weight('balanced', classes=unique_classes, y=y_train)
                        weights = torch.tensor(weights_np, dtype=torch.float32).to(self.device)
                        logger.info(f"Using balanced class weights: {weights_np.round(3)}")
                    else:
                         logger.warning("Only one class present in training data. Cannot compute balanced weights.")
                except Exception as e:
                    logger.warning(f"Could not compute class weights: {e}. Using uniform weights.")

            return nn.CrossEntropyLoss(weight=weights)

    def _prepare_dataloaders(self, X: pd.DataFrame, y: pd.Series,
                             X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None
                             ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Scales data and creates PyTorch DataLoaders."""
        logger.debug("Scaling data and creating DataLoaders...")
        # Scale features - fit scaler only on training data
        X_train_scaled = self.scaler.fit_transform(X.values) # Fit and transform train
        logger.info(f"Scaler fitted on training data. Mean: {self.scaler.mean_[:5].round(3)}..., Scale: {self.scaler.scale_[:5].round(3)}...")

        X_val_scaled = self.scaler.transform(X_val.values) if X_val is not None else None

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        # Target type depends on loss function
        is_ordinal = self.model_config.get('architecture', {}).get('ordinal_output', False)
        y_dtype = torch.float32 if is_ordinal else torch.long # Float for MSE, Long for CrossEntropy
        y_train_tensor = torch.tensor(y.values, dtype=y_dtype)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        batch_size = self.model_config.get('training', {}).get('batch_size', 64)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) # Drop last to avoid small batches

        val_loader = None
        if X_val_scaled is not None and y_val is not None:
            y_val_tensor = torch.tensor(y_val.values, dtype=y_dtype)
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            # Use larger batch size for validation as gradients aren't computed
            val_loader = DataLoader(val_dataset, batch_size=batch_size * 4, shuffle=False)
        else:
             logger.warning("No validation data provided. Early stopping and validation metrics will be unavailable.")

        logger.debug(f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader) if val_loader else 0}")
        return train_loader, val_loader

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """Train the Neural Network classifier."""
        start_time = time.time()
        self.feature_names_in_ = X.columns.tolist() # Store feature names

        # --- HPO Phase (if enabled) ---
        hpo_cfg = self.model_config.get('hyperparameter_optimization', {})
        hpo_enabled = hpo_cfg.get('enabled', False)

        if hpo_enabled:
             logger.info(f"Starting hyperparameter optimization for {self.model_name}...")
             if not _optuna_available:
                 logger.error("Optuna not available, cannot perform hyperparameter optimization. Please install optuna (`pip install optuna`).")
                 raise ImportError("Optuna is required for NN hyperparameter optimization but not installed.")
             try:
                 # Use provided validation set for HPO trials
                 if X_val is None or y_val is None:
                      raise ValueError("Validation data (X_val, y_val) is required for hyperparameter optimization.")
                 best_hpo_params = self.hyperparameter_optimize(X, y, X_val, y_val)
                 # Update self.model_config with the best params found for the final training run
                 logger.info(f"Updating model config with best HPO params for final training: {best_hpo_params}")
                 # This requires carefully merging best_hpo_params into self.model_config structure
                 arch_cfg = self.model_config.setdefault('architecture', {})
                 train_cfg = self.model_config.setdefault('training', {})
                 arch_cfg['hidden_layers'] = best_hpo_params.get('hidden_layers', arch_cfg.get('hidden_layers'))
                 arch_cfg['activation'] = best_hpo_params.get('activation', arch_cfg.get('activation'))
                 arch_cfg['dropout'] = best_hpo_params.get('dropout', arch_cfg.get('dropout'))
                 train_cfg['learning_rate'] = best_hpo_params.get('learning_rate', train_cfg.get('learning_rate'))
                 train_cfg['batch_size'] = best_hpo_params.get('batch_size', train_cfg.get('batch_size'))
                 train_cfg['optimizer'] = best_hpo_params.get('optimizer', train_cfg.get('optimizer'))

             except (NotImplementedError, ValueError, ImportError) as e:
                 logger.error(f"Hyperparameter optimization failed: {e}. Proceeding with default parameters from config.", exc_info=True)
                 # Continue with default parameters

        # --- Final Model Training ---
        logger.info(f"Starting final training for {self.model_name}...")
        train_loader, val_loader = self._prepare_dataloaders(X, y, X_val, y_val)

        # Initialize model, loss, optimizer based on potentially updated config
        self.model = DrWiggleNN(len(self.feature_names_in_), self.num_classes, self.model_config).to(self.device)
        criterion = self._get_loss_function(y.values) # Pass train labels for class weights
        optimizer = self._get_optimizer(self.model.parameters())

        # Optional: Learning rate scheduler
        scheduler = None
        lr_scheduler_cfg = self.model_config.get('training', {}).get('lr_scheduler', {}) # Use get for safety
        if lr_scheduler_cfg and lr_scheduler_cfg.get('enabled', False):
             scheduler_params = lr_scheduler_cfg
             scheduler = ReduceLROnPlateau(optimizer,
                                           mode='min' if self._is_loss_objective() else 'max',
                                           factor=scheduler_params.get('factor', 0.1),
                                           patience=scheduler_params.get('patience', 5), # Scheduler patience
                                           verbose=True)
             logger.info(f"Enabled ReduceLROnPlateau LR scheduler (factor={scheduler.factor}, patience={scheduler.patience}).")


        # Training loop config
        train_cfg = self.model_config.get('training', {})
        epochs = train_cfg.get('epochs', 100)
        early_stopping_enabled = train_cfg.get('early_stopping', True) and val_loader is not None
        patience = train_cfg.get('patience', 10) # Early stopping patience
        patience_counter = 0
        minimize_metric = self._is_loss_objective() # True if we monitor loss, False if accuracy

        # Reset history and best state before training loop
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        self.best_val_metric = float('inf') if minimize_metric else float('-inf')
        self.best_model_state_dict = None

        logger.info(f"Training {self.model_name} for {epochs} epochs on device {self.device}...")
        for epoch in range(epochs):
            self.model.train()
            epoch_train_loss = 0.0
            # Training Batches
            for batch_X, batch_y in progress_bar(train_loader, desc=f"Epoch {epoch+1}/{epochs} Train", leave=False):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)

                # Adjust target shape for MSELoss if needed
                if isinstance(criterion, nn.MSELoss): # Ordinal placeholder
                     loss = criterion(outputs, batch_y.unsqueeze(-1)) # Target needs shape (batch_size, 1)
                else: # Classification
                    loss = criterion(outputs, batch_y)

                loss.backward()
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_train_loss += loss.item() * batch_X.size(0) # Accumulate loss scaled by batch size

            avg_train_loss = epoch_train_loss / len(train_loader.dataset)
            self.history['train_loss'].append(avg_train_loss)

            # Validation Phase
            avg_val_loss = np.nan
            val_accuracy = np.nan
            current_val_metric = np.nan # Metric used for early stopping/best model saving

            if val_loader:
                 self.model.eval()
                 epoch_val_loss = 0.0
                 correct_preds = 0
                 total_preds = 0
                 with torch.no_grad():
                      for batch_X_val, batch_y_val in val_loader:
                           batch_X_val, batch_y_val = batch_X_val.to(self.device), batch_y_val.to(self.device)
                           outputs_val = self.model(batch_X_val)

                           # Calculate validation loss
                           if isinstance(criterion, nn.MSELoss): # Ordinal placeholder
                                loss_val = criterion(outputs_val, batch_y_val.unsqueeze(-1))
                           else: # Classification
                                loss_val = criterion(outputs_val, batch_y_val)
                           epoch_val_loss += loss_val.item() * batch_X_val.size(0)

                           # Calculate accuracy (adapt for ordinal placeholder)
                           if isinstance(criterion, nn.MSELoss):
                                # Simple accuracy for ordinal: check if rounded prediction matches target index
                                preds = torch.round(outputs_val.squeeze()).clamp(0, self.num_classes - 1).long()
                           else:
                                _, preds = torch.max(outputs_val, 1)

                           correct_preds += (preds == batch_y_val).sum().item()
                           total_preds += batch_y_val.size(0)

                 avg_val_loss = epoch_val_loss / len(val_loader.dataset)
                 val_accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
                 self.history['val_loss'].append(avg_val_loss)
                 self.history['val_accuracy'].append(val_accuracy)

                 # Determine the metric for comparison (loss or accuracy)
                 current_val_metric = avg_val_loss if minimize_metric else val_accuracy

                 log_msg = (f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.5f}, "
                            f"Val Loss: {avg_val_loss:.5f}, Val Acc: {val_accuracy:.4f}")

                 # LR Scheduler step
                 if scheduler:
                     scheduler.step(current_val_metric)
                     # Log current LR
                     current_lr = optimizer.param_groups[0]['lr']
                     log_msg += f", LR: {current_lr:.1e}"

                 logger.info(log_msg)

                 # Early Stopping & Best Model Saving Check
                 improvement = False
                 if np.isnan(current_val_metric):
                     logger.warning(f"Validation metric is NaN at epoch {epoch+1}. Cannot determine improvement.")
                 elif minimize_metric: # Lower is better (loss)
                     if current_val_metric < self.best_val_metric:
                         improvement = True
                 else: # Higher is better (accuracy)
                     if current_val_metric > self.best_val_metric:
                         improvement = True

                 if improvement:
                      self.best_val_metric = current_val_metric
                      patience_counter = 0
                      # Save the state dict of the best model found so far
                      self.best_model_state_dict = self.model.state_dict()
                      logger.debug(f"Validation metric improved to {self.best_val_metric:.4f}. Resetting patience. Saved best model state.")
                 else:
                      patience_counter += 1
                      logger.debug(f"Validation metric did not improve. Patience: {patience_counter}/{patience}")

                 if early_stopping_enabled and patience_counter >= patience:
                     logger.info(f"Early stopping triggered at epoch {epoch+1} due to lack of improvement for {patience} epochs.")
                     break # Exit training loop

            else: # No validation loader
                 self.history['val_loss'].append(np.nan)
                 self.history['val_accuracy'].append(np.nan)
                 logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.5f}")


        # After loop finishes (or breaks), load the best model state if available
        if self.best_model_state_dict:
            logger.info(f"Restoring model weights from epoch with best validation metric ({self.best_val_metric:.4f}).")
            self.model.load_state_dict(self.best_model_state_dict)
        else:
             # This can happen if no validation data was provided, or if training stopped at epoch 0,
             # or if the validation metric never improved.
             logger.warning("No best model state was saved during training. Using the final model state (which might not be optimal).")

        self._fitted = True
        logger.info(f"NN training finished in {time.time() - start_time:.2f} seconds.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate class predictions."""
        super().predict(X) # Checks fitted state and features
        if self.model is None: raise RuntimeError("Model is not initialized.")
        self.model.eval() # Set model to evaluation mode

        try:
             X_scaled = self.scaler.transform(X[self.feature_names_in_].values) # Scale using fitted scaler
        except NotFittedError:
             raise RuntimeError("Scaler has not been fitted. Call fit() before predicting.")
        except Exception as e:
             logger.error(f"Error scaling input data during prediction: {e}", exc_info=True)
             raise

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        predictions = []
        # Predict in batches to avoid OOM errors on large inputs
        # Use a potentially larger batch size for inference
        batch_size = self.model_config.get('training', {}).get('batch_size', 64) * 4
        pred_dataset = TensorDataset(X_tensor)
        # Disable num_workers for simple inference if causing issues
        pred_loader = DataLoader(pred_dataset, batch_size=batch_size, num_workers=0, pin_memory=False)


        with torch.no_grad():
            for batch_X_list in progress_bar(pred_loader, desc="Predicting", leave=False):
                 batch_X = batch_X_list[0].to(self.device) # DataLoader returns list/tuple, move to device
                 outputs = self.model(batch_X)
                 # Convert outputs to class indices
                 if self.model_config.get('architecture', {}).get('ordinal_output', False):
                     # Placeholder: Round output for ordinal prediction, clamp to valid range
                     preds = torch.round(outputs.squeeze()).clamp(0, self.num_classes - 1).long()
                 else:
                     # Standard classification: argmax
                     _, preds = torch.max(outputs, 1)
                 predictions.append(preds.cpu().numpy())

        return np.concatenate(predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate class probabilities."""
        super().predict_proba(X) # Checks fitted state and features
        if self.model is None: raise RuntimeError("Model is not initialized.")
        self.model.eval() # Set model to evaluation mode

        try:
             X_scaled = self.scaler.transform(X[self.feature_names_in_].values)
        except NotFittedError:
             raise RuntimeError("Scaler has not been fitted. Call fit() before predicting probabilities.")
        except Exception as e:
             logger.error(f"Error scaling input data during probability prediction: {e}", exc_info=True)
             raise

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        probabilities = []
        # Predict in batches
        batch_size = self.model_config.get('training', {}).get('batch_size', 64) * 4
        pred_dataset = TensorDataset(X_tensor)
        pred_loader = DataLoader(pred_dataset, batch_size=batch_size, num_workers=0, pin_memory=False)


        with torch.no_grad():
            for batch_X_list in progress_bar(pred_loader, desc="Predicting Probs", leave=False):
                batch_X = batch_X_list[0].to(self.device)
                outputs = self.model(batch_X)

                # Calculate probabilities
                if self.model_config.get('architecture', {}).get('ordinal_output', False):
                    # Probabilities are not well-defined for the MSE ordinal placeholder.
                    # Return uniform probabilities or raise error? Returning uniform for now.
                    logger.warning("Probability prediction not well-defined for current ordinal placeholder. Returning uniform distribution.")
                    # Need to ensure output shape is correct for broadcasting
                    uniform_prob = 1.0 / self.num_classes
                    probs = torch.full((outputs.shape[0], self.num_classes), uniform_prob, device=self.device, dtype=torch.float32)
                else:
                    # Apply Softmax for standard classification probabilities
                    probs = torch.softmax(outputs, dim=1)

                probabilities.append(probs.cpu().numpy())

        return np.concatenate(probabilities)

    def save(self, path: str):
        """Save the trained model state, scaler, and config."""
        super().save(path) # Checks fitted, creates dir
        if not hasattr(self, 'scaler') or not hasattr(self.scaler, 'mean_'):
             logger.warning(f"Scaler not found or not fitted for model '{self.model_name}'. Saving model without scaler state.")
             scaler_state = None
        else:
             scaler_state = self.scaler

        if self.model is None:
            logger.warning(f"Model object is None for '{self.model_name}'. Cannot save model state dict.")
            model_state_dict_to_save = None
        else:
            model_state_dict_to_save = self.model.state_dict()


        state = {
            'model_state_dict': model_state_dict_to_save,
            'scaler_state': scaler_state, # Save the fitted scaler instance
            'feature_names_in_': self.feature_names_in_,
            'config': self.config, # Save config used for this instance
            'model_config': self.model_config, # Save specific config for this model
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'history': self.history,
            'fitted': self._fitted
        }
        try:
            # Use torch.save for potentially large tensors in state_dict
            torch.save(state, path)
            logger.info(f"NeuralNetwork model state saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save NeuralNetwork model state to {path}: {e}", exc_info=True)
            raise IOError(f"Could not save NN model state to {path}") from e

    @classmethod
    def load(cls, path: str, config: Optional[Dict[str, Any]] = None) -> 'NeuralNetworkClassifier':
        """Load a trained model state from disk."""
        # Base class load method only checks file existence in this implementation
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        logger.info(f"Loading model '{cls.__name__}' from {path}...")

        # Determine device based on availability during load time, respecting config if provided
        runtime_config_system = config.get("system", {}) if config else {}
        gpu_enabled_cfg = runtime_config_system.get("gpu_enabled", "auto")
        force_cpu = gpu_enabled_cfg == False

        if not force_cpu and torch.cuda.is_available(): device = torch.device("cuda")
        elif not force_cpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = torch.device("mps")
        else: device = torch.device("cpu")

        logger.info(f"Loading NeuralNetwork state from {path} to device {device}")
        try:
             # Load state dict using torch.load, mapping to the determined device
             state = torch.load(path, map_location=device)
        except Exception as e:
            logger.error(f"Failed to load torch state from {path}: {e}", exc_info=True)
            raise IOError(f"Could not load NN model state from {path}") from e

        # Validate loaded state
        required_keys = ['model_state_dict', 'scaler_state', 'feature_names_in_', 'model_config', 'model_name', 'num_classes', 'fitted']
        if not all(key in state for key in required_keys):
             # Allow for missing history in older versions
             if not all(key in state for key in required_keys if key != 'history'):
                raise ValueError(f"Loaded NN state from {path} is missing required keys. Found: {list(state.keys())}")

        # Use loaded config if available, otherwise fall back to provided runtime config
        load_config = state.get('config', config)
        if load_config is None:
             raise ValueError("Cannot load NN model: No configuration found in saved state or provided at runtime.")

        # Re-instantiate the class
        instance = cls(config=load_config, model_name=state['model_name'])

        # Restore state
        instance.scaler = state['scaler_state']
        instance.feature_names_in_ = state['feature_names_in_']
        instance.num_classes = state['num_classes']
        instance.history = state.get('history', {'train_loss': [], 'val_loss': [], 'val_accuracy': []}) # Handle older saves
        instance._fitted = state['fitted']
        instance.device = device # Set device on loaded instance

        # Re-create model architecture based on loaded config and load state dict
        if state['model_state_dict']:
             input_dim = len(instance.feature_names_in_) if instance.feature_names_in_ else None
             if input_dim is None and instance.scaler and hasattr(instance.scaler, 'n_features_in_'):
                   input_dim = instance.scaler.n_features_in_
             if input_dim is None:
                   raise ValueError("Cannot determine input dimension for loading NN model (missing feature names and scaler info).")

             instance.model = DrWiggleNN(input_dim, instance.num_classes, instance.model_config).to(instance.device)
             instance.model.load_state_dict(state['model_state_dict'])
             instance.model.eval() # Set to evaluation mode
             logger.info("Model architecture recreated and state dict loaded.")
        else:
             logger.warning(f"Loaded state for '{instance.model_name}' has no model_state_dict. Model not loaded.")
             instance.model = None
             instance._fitted = False # Mark as not fitted if model state is missing

        # Validate scaler
        if instance.scaler and not hasattr(instance.scaler, 'mean_'):
             logger.warning(f"Loaded scaler for model '{instance.model_name}' appears not to be fitted.")
             # Should we mark instance._fitted as False? Or assume model can run without scaler? Risky.
             instance._fitted = False # If scaler missing/unfitted, cannot reliably predict

        if not instance._fitted:
             logger.warning(f"Loaded model '{instance.model_name}' from {path} is marked as not fitted (check model state and scaler).")

        logger.info(f"Model '{instance.model_name}' loaded successfully.")
        return instance


    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Feature importance for NNs requires advanced methods (SHAP, Captum, Permutation). Not implemented."""
        logger.warning(f"Feature importance calculation is not implemented for '{self.__class__.__name__}'. Requires methods like SHAP or Permutation Importance.")
        return None

    def hyperparameter_optimize(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
         """Perform hyperparameter optimization using Optuna."""
         hpo_cfg = self.model_config.get('hyperparameter_optimization', {})
         if not hpo_cfg.get('enabled', False):
              raise NotImplementedError(f"Hyperparameter optimization not enabled in config for '{self.model_name}'.")
         if not _optuna_available:
              raise ImportError("Optuna is required for NN HPO but not installed (`pip install optuna`).")

         method = hpo_cfg.get('method', 'random').lower() # TODO: Add support for TPE etc.
         n_trials = hpo_cfg.get('trials', 20)
         param_space_cfg = hpo_cfg.get('parameters', {})
         objective_metric = hpo_cfg.get('objective_metric', 'val_loss').lower()
         hpo_epochs = hpo_cfg.get('epochs_per_trial', 50) # Limit epochs per trial
         hpo_patience = hpo_cfg.get('patience_per_trial', 5) # Shorter patience for HPO

         if not param_space_cfg:
              raise ValueError(f"Hyperparameter optimization enabled but no 'parameters' space defined in config for '{self.model_name}'.")

         logger.info(f"Starting Optuna HPO for '{self.model_name}' ({method} search, {n_trials} trials, objective: {objective_metric}, {hpo_epochs} epochs/trial)...")

         # --- Pre-prepare DataLoaders once outside the objective function ---
         # This saves repeated scaling and tensor conversion in each trial
         logger.debug("Preparing DataLoaders for HPO trials...")
         # Fit scaler on the full training set passed to HPO
         X_train_scaled_hpo = self.scaler.fit_transform(X_train.values)
         X_val_scaled_hpo = self.scaler.transform(X_val.values)
         is_ordinal_hpo = self.model_config.get('architecture', {}).get('ordinal_output', False)
         y_dtype_hpo = torch.float32 if is_ordinal_hpo else torch.long
         y_train_tensor_hpo = torch.tensor(y_train.values, dtype=y_dtype_hpo)
         y_val_tensor_hpo = torch.tensor(y_val.values, dtype=y_dtype_hpo)

         logger.debug("Data scaling and tensor conversion complete for HPO.")

         # --- Objective Function for Optuna ---
         def objective(trial: optuna.Trial) -> float:
             # Suggest hyperparameters based on param_space_cfg
             trial_arch_params = {}
             trial_train_params = {}

             # Architecture Params
             hs_choices = param_space_cfg.get('hidden_layers')
             trial_arch_params['hidden_layers'] = trial.suggest_categorical('hidden_layers', hs_choices) if hs_choices else [64, 32]
             act_choices = param_space_cfg.get('activation')
             trial_arch_params['activation'] = trial.suggest_categorical('activation', act_choices) if act_choices else 'relu'
             drp_range = param_space_cfg.get('dropout')
             trial_arch_params['dropout'] = trial.suggest_float('dropout', drp_range[0], drp_range[1]) if drp_range and len(drp_range)==2 else trial.suggest_float('dropout', 0.1, 0.5) # Default range

             # Training Params
             lr_range = param_space_cfg.get('learning_rate')
             trial_train_params['learning_rate'] = trial.suggest_float('learning_rate', lr_range[0], lr_range[1], log=True) if lr_range and len(lr_range)==2 else trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
             bs_choices = param_space_cfg.get('batch_size')
             trial_train_params['batch_size'] = trial.suggest_categorical('batch_size', bs_choices) if bs_choices else 64
             opt_choices = param_space_cfg.get('optimizer')
             trial_train_params['optimizer'] = trial.suggest_categorical('optimizer', opt_choices) if opt_choices else 'adam'

             # Create temporary config for this trial by overlaying suggestions
             # Important: Create deep copies to avoid modifying original config
             trial_model_config = joblib.load(joblib.dump(self.model_config, 'temp_config.joblib')) # Deep copy via dump/load hack or use copy.deepcopy
             trial_model_config['architecture'].update(trial_arch_params)
             trial_model_config['training'].update(trial_train_params)
             # Use limited epochs/patience for HPO trial
             trial_model_config['training']['epochs'] = hpo_epochs
             trial_model_config['training']['patience'] = hpo_patience

             # Create a temporary config object for the trial model instance
             trial_full_config = joblib.load(joblib.dump(self.config, 'temp_config.joblib'))
             trial_full_config['models'][self.model_name] = trial_model_config # Update model section
             os.remove('temp_config.joblib') # Clean up temporary file


             logger.debug(f"Trial {trial.number}: Params: {trial.params}")

             # Instantiate and train a temporary model for this trial
             try:
                 # --- Create DataLoaders INSIDE trial using pre-scaled data ---
                 X_train_tensor_trial = torch.tensor(X_train_scaled_hpo, dtype=torch.float32)
                 X_val_tensor_trial = torch.tensor(X_val_scaled_hpo, dtype=torch.float32)
                 train_dataset_trial = TensorDataset(X_train_tensor_trial, y_train_tensor_hpo)
                 val_dataset_trial = TensorDataset(X_val_tensor_trial, y_val_tensor_hpo)
                 trial_batch_size = trial_train_params['batch_size']
                 train_loader_trial = DataLoader(train_dataset_trial, batch_size=trial_batch_size, shuffle=True, drop_last=True)
                 val_loader_trial = DataLoader(val_dataset_trial, batch_size=trial_batch_size * 4, shuffle=False)
                 logger.debug(f"Trial {trial.number}: Created DataLoaders.")

                 # Instantiate model using trial config
                 temp_model = DrWiggleNN(len(self.feature_names_in_), self.num_classes, trial_model_config).to(self.device)
                 temp_criterion = self._get_loss_function(y_train.values) # Use original y_train for weights
                 temp_optimizer = self._get_optimizer(temp_model.parameters()) # Optimizer based on trial params

                 # --- Simplified Training Loop for HPO Trial ---
                 trial_history = {'val_loss': [], 'val_accuracy': []}
                 best_trial_metric = float('inf') if self._is_loss_objective() else float('-inf')
                 trial_patience_counter = 0

                 for epoch in range(hpo_epochs):
                     temp_model.train()
                     for batch_X, batch_y in train_loader_trial: # Use trial loader
                          batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                          temp_optimizer.zero_grad()
                          outputs = temp_model(batch_X)
                          if isinstance(temp_criterion, nn.MSELoss): loss = temp_criterion(outputs, batch_y.unsqueeze(-1))
                          else: loss = temp_criterion(outputs, batch_y)
                          loss.backward()
                          temp_optimizer.step()

                     # Simplified Validation
                     temp_model.eval()
                     epoch_val_loss = 0.0
                     correct_preds = 0
                     total_preds = 0
                     with torch.no_grad():
                          for batch_X_val, batch_y_val in val_loader_trial: # Use trial loader
                               batch_X_val, batch_y_val = batch_X_val.to(self.device), batch_y_val.to(self.device)
                               outputs_val = temp_model(batch_X_val)
                               if isinstance(temp_criterion, nn.MSELoss):
                                    loss_val = temp_criterion(outputs_val, batch_y_val.unsqueeze(-1))
                                    preds = torch.round(outputs_val.squeeze()).clamp(0, self.num_classes - 1).long()
                               else:
                                    loss_val = temp_criterion(outputs_val, batch_y_val)
                                    _, preds = torch.max(outputs_val, 1)
                                epoch_val_loss += loss_val.item() * batch_X_val.size(0)
                                correct_preds += (preds == batch_y_val).sum().item()
                                total_preds += batch_y_val.size(0)

                     avg_val_loss = epoch_val_loss / len(val_loader_trial.dataset)
                     val_accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
                     trial_history['val_loss'].append(avg_val_loss)
                     trial_history['val_accuracy'].append(val_accuracy)
                     current_trial_metric = avg_val_loss if self._is_loss_objective() else val_accuracy

                     # Pruning & Early stopping check within trial
                     trial.report(current_trial_metric, epoch)
                     if trial.should_prune():
                          logger.debug(f"Trial {trial.number} pruned at epoch {epoch+1}.")
                          raise optuna.TrialPruned()

                     # Check for improvement
                     improved = False
                     if np.isnan(current_trial_metric): continue # Skip comparison if NaN
                     if self._is_loss_objective():
                          if current_trial_metric < best_trial_metric: improved = True
                     else:
                          if current_trial_metric > best_trial_metric: improved = True

                     if improved:
                          best_trial_metric = current_trial_metric
                          trial_patience_counter = 0
                     else:
                          trial_patience_counter += 1

                     if trial_patience_counter >= hpo_patience:
                          logger.debug(f"Trial {trial.number} stopped early at epoch {epoch+1} due to patience.")
                          break # Stop trial early

                 # --- Return the best metric achieved in this trial ---
                 final_trial_result = best_trial_metric # Return the best metric found during the trial
                 logger.debug(f"Trial {trial.number}: Result ({objective_metric}) = {final_trial_result:.5f}")
                 return final_trial_result

             except optuna.TrialPruned:
                 raise # Re-raise prune signals for Optuna
             except Exception as e:
                  logger.warning(f"Trial {trial.number} failed with error: {e}. Returning worst possible score.", exc_info=True)
                  # Return worst score depending on optimization direction
                  return float('inf') if self._is_loss_objective() else float('-inf')


         # --- Run Optuna Study ---
         study_direction = 'minimize' if self._is_loss_objective() else 'maximize'
         study = optuna.create_study(direction=study_direction)
         optimize_kwargs = {'n_trials': n_trials}
         if _optuna_supports_progbar: optimize_kwargs['show_progress_bar'] = True
         study.optimize(objective, **optimize_kwargs)

         best_params = study.best_params
         best_value = study.best_value
         logger.info(f"Optuna HPO complete. Best validation {objective_metric}: {best_value:.5f}")
         logger.info(f"Best parameters found: {best_params}")

         # Return the best parameters found
         return best_params
EOF

echo "Creating utils/metrics.py..."
cat << 'EOF' > drwiggle/utils/metrics.py
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
EOF

echo "Creating utils/pdb_tools.py..."
cat << 'EOF' > drwiggle/utils/pdb_tools.py
import logging
import os
import re
import warnings
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

# Biopython imports
try:
    from Bio.PDB import PDBParser, PDBIO, Select, Polypeptide
    from Bio.PDB.DSSP import DSSP
    from Bio.PDB.exceptions import PDBException
    from Bio.PDB.PDBList import PDBList
    from Bio.SeqUtils import seq1 # To convert 3-letter AA code to 1-letter
    _biopython_available = True
except ImportError:
    logging.getLogger(__name__).critical("BioPython not found. Please install it (`pip install biopython`) to use PDB processing features.")
    # Define dummy classes/functions to avoid errors if module is imported but BP not installed
    class PDBParser: pass
    class PDBIO: pass
    class Select: pass
    class Polypeptide: pass
    class DSSP: pass
    class PDBException(Exception): pass
    class PDBList: pass
    def seq1(res): return 'X'
    _biopython_available = False

# Local imports
from drwiggle.config import get_pdb_config, get_pdb_feature_config
from drwiggle.utils.helpers import ensure_dir

logger = logging.getLogger(__name__)

# --- PDB Parsing and Feature Extraction ---

def fetch_pdb(pdb_id: str, cache_dir: str) -> Optional[str]:
    """
    Downloads a PDB file if not already cached.

    Args:
        pdb_id: The 4-character PDB ID.
        cache_dir: The directory to store/retrieve PDB files.

    Returns:
        The path to the cached PDB file (format .pdb), or None if download fails.
    """
    if not _biopython_available:
         logger.error("BioPython PDBList not available for fetching PDB files.")
         return None

    ensure_dir(cache_dir)
    pdb_list = PDBList(pdb=cache_dir, obsolete_pdb=cache_dir, verbose=False)
    # Explicitly request pdb format, adjust filename handling
    try:
        # retrieve_pdb_file returns the path it *would* have if downloaded/cached
        expected_path = pdb_list.retrieve_pdb_file(pdb_id, pdir=cache_dir, file_format='pdb')

        # Check if the file actually exists after retrieve_pdb_file call
        if os.path.exists(expected_path):
            logger.info(f"PDB file for {pdb_id} found/downloaded at: {expected_path}")
            return expected_path
        else:
             # Sometimes PDBList doesn't error but fails to download
             logger.error(f"Failed to retrieve PDB file for {pdb_id} (expected path: {expected_path}). Check ID and network.")
             return None

    except Exception as e:
        logger.error(f"Error retrieving PDB file for {pdb_id}: {e}", exc_info=True)
        return None

def parse_pdb(pdb_path_or_id: str, pdb_config: Dict[str, Any]) -> Optional[Any]:
    """
    Parses a PDB file using BioPython's PDBParser. Handles fetching if ID is given.

    Args:
        pdb_path_or_id: Path to the PDB file or a 4-character PDB ID.
        pdb_config: PDB configuration dictionary (must contain 'pdb_cache_dir').

    Returns:
        Bio.PDB Model object (the first model found), or None if parsing/fetching fails.
    """
    if not _biopython_available:
         logger.error("BioPython PDBParser not available for parsing PDB files.")
         return None

    pdb_id_pattern = re.compile(r"^[a-zA-Z0-9]{4}$")
    pdb_path = None
    structure_id = "structure" # Default ID for the structure object

    if os.path.isfile(pdb_path_or_id):
        pdb_path = os.path.abspath(pdb_path_or_id)
        structure_id = os.path.splitext(os.path.basename(pdb_path))[0] # Use filename stem as ID
        logger.info(f"Parsing local PDB file: {pdb_path}")
    elif pdb_id_pattern.match(pdb_path_or_id):
        pdb_id = pdb_path_or_id.upper()
        structure_id = pdb_id # Use PDB ID as structure ID
        cache_dir = pdb_config.get('pdb_cache_dir')
        if not cache_dir:
             logger.error("pdb_cache_dir not specified in config. Cannot fetch PDB ID.")
             return None
        logger.info(f"Attempting to fetch PDB ID: {pdb_id} using cache: {cache_dir}")
        pdb_path = fetch_pdb(pdb_id, cache_dir)
        if not pdb_path: return None # Fetch failed
    else:
        logger.error(f"Invalid PDB input: '{pdb_path_or_id}'. Must be a valid file path or 4-character PDB ID.")
        return None

    parser = PDBParser(QUIET=True, STRUCTURE_BUILDER=Polypeptide.PolypeptideBuilder()) # Use builder for phi/psi
    try:
        structure = parser.get_structure(structure_id, pdb_path)
        logger.info(f"Successfully parsed PDB structure '{structure.id}'. Models: {len(structure)}")
        if len(structure) > 1:
             logger.warning(f"PDB file contains multiple models ({len(structure)}). Using only the first model (ID: {structure[0].id}).")
        if len(structure) == 0:
            logger.error(f"No models found in PDB structure '{structure.id}'. Cannot proceed.")
            return None
        return structure[0] # Return only the first model
    except PDBException as e:
        logger.error(f"Bio.PDB parsing error for {pdb_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing PDB file {pdb_path}: {e}", exc_info=True)
        return None


def extract_pdb_features(
    structure_model: Any, # Should be a Bio.PDB Model object
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Extracts features like B-factor, SS, ACC, Dihedrals from a Bio.PDB Model.

    Args:
        structure_model: The Bio.PDB Model object (typically structure[0]).
        config: The main configuration dictionary.

    Returns:
        DataFrame containing extracted features per residue.
    """
    if not _biopython_available:
        logger.error("BioPython not available, cannot extract PDB features.")
        return pd.DataFrame()

    pdb_config = get_pdb_config(config)
    feature_flags = get_pdb_feature_config(config)
    model_id = structure_model.id
    pdb_structure_id = structure_model.get_parent().id # Get the overall structure ID
    data = []

    # --- Run DSSP if needed ---
    dssp_results = None
    dssp_path = pdb_config.get('dssp_path') # Path to executable
    needs_dssp = feature_flags.get('secondary_structure') or feature_flags.get('solvent_accessibility')

    if needs_dssp:
        # DSSP requires a file path. Save the model temporarily.
        # Using a unique temp name based on structure and model ID
        temp_pdb_dir = os.path.join(pdb_config.get("pdb_cache_dir", "."), "temp") # Save in cache/temp
        ensure_dir(temp_pdb_dir)
        temp_pdb_path = os.path.join(temp_pdb_dir, f"_temp_{pdb_structure_id}_model_{model_id}.pdb")

        io = PDBIO()
        io.set_structure(structure_model)
        io.save(temp_pdb_path)
        logger.debug(f"Temporarily saved model {model_id} to {temp_pdb_path} for DSSP.")

        try:
            logger.info(f"Running DSSP (using path: {dssp_path or 'system PATH'})...")
            # Pass model object AND file path to DSSP constructor
            dssp_results = DSSP(structure_model, temp_pdb_path, dssp=dssp_path)
            logger.info(f"DSSP calculation successful for {len(dssp_results)} residues.")
        except FileNotFoundError as e:
             # Check if dssp_path was specified or if it failed from PATH search
             search_location = f"specified path '{dssp_path}'" if dssp_path else "system PATH"
             logger.error(f"DSSP executable not found at {search_location}. Cannot calculate SS/ACC. Error: {e}")
             logger.error("Please install DSSP (e.g., `sudo apt install dssp` or `conda install dssp`) "
                          "and ensure it's in your PATH, or set 'pdb.dssp_path' in config.")
             dssp_results = None # Ensure it's None if failed
        except PDBException as e: # Catch DSSP execution errors (e.g., invalid PDB for DSSP)
             logger.error(f"DSSP calculation failed for {temp_pdb_path}: {e}")
             dssp_results = None
        except Exception as e: # Catch other unexpected errors
             logger.error(f"Unexpected error running DSSP: {e}", exc_info=True)
             dssp_results = None
        finally:
             # Clean up temporary PDB file
             if os.path.exists(temp_pdb_path):
                  try: os.remove(temp_pdb_path)
                  except OSError: logger.warning(f"Could not remove temporary PDB file: {temp_pdb_path}")

    # --- Iterate through residues and extract features ---
    logger.info(f"Extracting features for Model ID: {model_id} of Structure: {pdb_structure_id}")
    residue_counter = 0
    phi_psi_list = [] # Store calculated dihedrals if requested early
    if feature_flags.get('dihedral_angles'):
        # Calculate dihedrals for all polypeptides in the model first
        for chain in structure_model:
            try:
                # PolypeptideBuilder should have been used during parsing for phi/psi
                polypeptides = chain.get_list() # Access polypeptides built during parsing
                for pp in polypeptides:
                    if isinstance(pp, Polypeptide.Polypeptide): # Check it's a polypeptide object
                        phi_psi_list.extend(pp.get_phi_psi_list())
                    else:
                        logger.warning(f"Unexpected object in chain list: {type(pp)}. Skipping dihedral calculation for this object.")
            except Exception as e:
                 logger.error(f"Error getting phi/psi list for chain {chain.id}: {e}", exc_info=True)
                 # Continue processing other chains/features

    # Create a map for faster dihedral lookup: (chain_id, res_id_tuple) -> (phi, psi)
    phi_psi_map = {}
    if phi_psi_list:
        logger.debug(f"Mapping {len(phi_psi_list)} calculated phi/psi pairs to residues...")
        res_index = 0
        for chain in structure_model:
            for residue in chain.get_residues():
                # Important: Check if residue corresponds to an entry in phi_psi_list
                # This relies on the list being in the same order as residues iterated here. Risky.
                # It's safer to recalculate per residue if builder wasn't used or order is uncertain.
                # Let's recalculate on the fly for robustness, though slower.
                pass # Recalculate below instead of using precomputed list

    for chain in structure_model:
        chain_id = chain.id
        for residue in chain.get_residues():
            res_id_tuple = residue.get_id() # tuple: (hetflag, resid, icode)
            resname = residue.get_resname()

            # Skip HETATMs and non-standard residues
            # Check hetflag (' ' means standard residue atom)
            if res_id_tuple[0] != ' ':
                continue
            # Check if it's a standard amino acid using Polypeptide module
            try:
                is_standard_aa = Polypeptide.is_aa(residue, standard=True)
            except Exception: # Handle cases where is_aa might fail on modified residues
                is_standard_aa = False
            if not is_standard_aa:
                continue

            res_seq_id = res_id_tuple[1] # Residue sequence number
            res_icode = res_id_tuple[2].strip() # Insertion code (remove whitespace)

            residue_features = {
                # Use the main structure ID (e.g., PDB ID) as the domain identifier
                "domain_id": pdb_structure_id,
                "chain_id": chain_id,
                "resid": res_seq_id,
                # Only include icode if it's not empty
                **({"icode": res_icode} if res_icode else {}),
                "resname": resname,
            }
            residue_counter += 1

            # Extract B-factor (average over CA atom if present, else backbone, else all non-H)
            if feature_flags.get('b_factor'):
                ca_atom = residue.get("CA")
                if ca_atom:
                    bfactors = [ca_atom.get_bfactor()]
                else: # Fallback to backbone or all heavy atoms
                     backbone_atoms = ['N', 'CA', 'C', 'O']
                     bfactors = [atom.get_bfactor() for atom_name, atom in residue.items() if atom_name in backbone_atoms]
                     if not bfactors: # If still no backbone, use all heavy atoms
                          bfactors = [atom.get_bfactor() for atom in residue if atom.element != 'H']
                residue_features['b_factor'] = np.mean(bfactors) if bfactors else 0.0

            # Extract DSSP features (SS and ACC)
            ss = '-' # Default DSSP code for unknown/loop
            rsa = np.nan # Default accessibility
            if dssp_results:
                 # DSSP key uses the residue ID tuple directly (including hetflag, resid, icode)
                 dssp_key = (chain_id, res_id_tuple)
                 if dssp_key in dssp_results:
                      dssp_data = dssp_results[dssp_key]
                      # Index 2 is SS code, Index 3 is relative accessibility (RSA)
                      if feature_flags.get('secondary_structure'):
                           ss = dssp_data[2]
                      if feature_flags.get('solvent_accessibility'):
                           rsa = dssp_data[3]
                 else:
                      # Log only once per run if residues are missing to avoid spam
                      if not hasattr(extract_pdb_features, "_dssp_missing_logged"):
                           logger.warning(f"Residue {chain_id}:{res_id_tuple} not found in DSSP results. DSSP might skip residues. Subsequent warnings suppressed.")
                           extract_pdb_features._dssp_missing_logged = True


            residue_features['dssp'] = ss
            residue_features['relative_accessibility'] = rsa if not pd.isna(rsa) else None

            # Extract Dihedral angles (Phi, Psi)
            if feature_flags.get('dihedral_angles'):
                 phi = None
                 psi = None
                 try:
                      # Recalculate phi/psi using Polypeptide methods for robustness
                      phi = Polypeptide.calc_phi(residue)
                      psi = Polypeptide.calc_psi(residue)
                 except Exception as e:
                      # Dihedrals might be undefined at termini or chain breaks
                      logger.debug(f"Could not calculate phi/psi for {chain_id}:{res_id_tuple}: {e}")

                 # Convert radians from Biopython to degrees, handle None
                 residue_features['phi'] = np.degrees(phi) if phi is not None else None
                 residue_features['psi'] = np.degrees(psi) if psi is not None else None

            data.append(residue_features)

    # Reset DSSP logging flag for next call
    if hasattr(extract_pdb_features, "_dssp_missing_logged"):
        del extract_pdb_features._dssp_missing_logged

    df = pd.DataFrame(data)
    logger.info(f"Extracted features for {residue_counter} standard residues across {len(structure_model)} chains.")

    # --- Post-processing (e.g., calculate core/exterior) ---
    # This often requires more complex calculations (e.g., SASA, neighbor counts)
    if feature_flags.get('core_exterior_encoded'):
         logger.warning("Feature 'core_exterior_encoded' requested, but calculation logic is not implemented. Column will be missing or filled with UNK.")
         # Placeholder: add column if requested, needs actual calculation logic
         if 'relative_accessibility' in df.columns and not df['relative_accessibility'].isnull().all():
             # Example simple classification based on RSA threshold (e.g., 0.2)
             threshold = 0.20
             df['core_exterior'] = df['relative_accessibility'].apply(lambda x: 'SURFACE' if pd.notna(x) and x >= threshold else 'CORE')
             logger.info(f"Assigned 'core_exterior' based on RSA threshold ({threshold}).")
         else:
              df['core_exterior'] = 'UNK' # Add placeholder column if RSA not available

    # Ensure required columns exist even if feature extraction failed partially
    expected_cols = ['domain_id', 'chain_id', 'resid']
    if feature_flags.get('b_factor'): expected_cols.append('b_factor')
    if feature_flags.get('secondary_structure'): expected_cols.append('dssp')
    if feature_flags.get('solvent_accessibility'): expected_cols.append('relative_accessibility')
    if feature_flags.get('dihedral_angles'): expected_cols.extend(['phi', 'psi'])
    if feature_flags.get('core_exterior_encoded'): expected_cols.append('core_exterior')

    for col in expected_cols:
        if col not in df.columns:
            logger.warning(f"Expected feature column '{col}' not found after extraction. Adding column with NaN/defaults.")
            default_val = 0.0 if col == 'b_factor' else ('-' if col == 'dssp' else ('UNK' if col == 'core_exterior' else np.nan))
            df[col] = default_val

    return df

# --- PDB Visualization/Output ---

class ColorByFlexibilitySelect(Select):
    """Bio.PDB Selector to set B-factor based on predicted flexibility class."""
    def __init__(self, predictions_map: Dict[Tuple[str, int], int], default_b: float = 20.0):
        """
        Args:
            predictions_map: Dictionary mapping (chain_id, resid) to predicted_class.
            default_b: B-factor value for residues not in the predictions map.
        """
        self.predictions = predictions_map
        self.default_b = default_b
        # Map class index (0, 1, 2, 3, 4) to representative B-factor values (e.g., low B for rigid)
        # Higher B-factor usually means more flexible/disordered
        # Make this mapping potentially configurable? For now, hardcoded.
        self.class_to_bfactor = {
            0: 10.0,  # Very Rigid
            1: 25.0,  # Rigid
            2: 40.0,  # Moderately Flexible
            3: 60.0,  # Flexible
            4: 80.0,  # Very Flexible
        }
        # Add mapping for potential classes beyond 4 if num_classes > 5
        max_class = max(self.class_to_bfactor.keys())
        max_b = self.class_to_bfactor[max_class]
        for i in range(max_class + 1, 10): # Add up to class 9 as example
            self.class_to_bfactor[i] = max_b + (i - max_class) * 15.0 # Increment B-factor

        logger.debug(f"B-factor mapping for coloring: {self.class_to_bfactor}")

    def accept_atom(self, atom) -> int:
        """Accepts the atom and sets its B-factor based on prediction."""
        residue = atom.get_parent()
        chain = residue.get_parent()
        res_id_tuple = residue.get_id() # (hetflag, resid, icode)

        # Only modify standard AA residues' atoms
        if res_id_tuple[0] == ' ' and Polypeptide.is_aa(residue.get_resname(), standard=True):
            chain_id = chain.id
            res_seq_id = res_id_tuple[1] # Numeric residue ID
            # Key uses only chain and numeric resid for simplicity (ignores icode for matching)
            pred_key = (chain_id, res_seq_id)

            predicted_class = self.predictions.get(pred_key)
            if predicted_class is not None:
                 b_factor_val = self.class_to_bfactor.get(predicted_class, self.default_b)
                 atom.set_bfactor(float(b_factor_val)) # Ensure float B-factor
            else:
                 # Residue not found in predictions map
                 atom.set_bfactor(self.default_b)
                 # Log only once if residues are missing from the map
                 if not hasattr(ColorByFlexibilitySelect, "_missing_logged"):
                      logger.warning(f"Residue {chain_id}:{res_seq_id} not in prediction map. Setting B-factor to default {self.default_b}. Subsequent warnings suppressed.")
                      ColorByFlexibilitySelect._missing_logged = True
        else:
             # Keep original B-factor for HETATMs, non-standard residues, etc.
             pass

        return 1 # Keep the atom in the output


def color_pdb_by_flexibility(
    structure_model: Any, # Bio.PDB Model object
    predictions_df: pd.DataFrame, # Must contain 'chain_id', 'resid', 'predicted_class'
    output_pdb_path: str
):
    """
    Creates a new PDB file where the B-factor column reflects the predicted flexibility class.

    Args:
        structure_model: The Bio.PDB Model object to modify.
        predictions_df: DataFrame with prediction results.
        output_pdb_path: Path to save the colored PDB file.
    """
    if not _biopython_available:
        logger.error("BioPython not available. Cannot create colored PDB.")
        return

    logger.info(f"Generating colored PDB file (using B-factor column): {output_pdb_path}")

    # Create mapping for quick lookup: (chain_id, resid) -> predicted_class
    required_cols = ['chain_id', 'resid', 'predicted_class']
    if not all(col in predictions_df.columns for col in required_cols):
         logger.error(f"Predictions DataFrame must contain columns: {required_cols}. Found: {predictions_df.columns.tolist()}")
         return
    try:
        # Ensure 'resid' is integer for matching PDB residue IDs
        predictions_df['resid'] = predictions_df['resid'].astype(int)
        pred_map = predictions_df.set_index(['chain_id', 'resid'])['predicted_class'].to_dict()
    except KeyError:
        # This error is redundant due to the check above but kept as safeguard
        logger.error("Failed to create prediction map from DataFrame columns.")
        return
    except Exception as e:
         logger.error(f"Error creating prediction map: {e}", exc_info=True)
         return


    # Create PDBIO object and save with the custom selector
    io = PDBIO()
    io.set_structure(structure_model)
    ensure_dir(os.path.dirname(output_pdb_path))

    # Reset the logging flag before saving
    if hasattr(ColorByFlexibilitySelect, "_missing_logged"):
        del ColorByFlexibilitySelect._missing_logged

    try:
        # Use a default B=20 for residues not found in predictions
        io.save(output_pdb_path, select=ColorByFlexibilitySelect(pred_map, default_b=20.0))
        logger.info(f"Colored PDB saved successfully to {output_pdb_path}")
    except Exception as e:
        logger.error(f"Failed to save colored PDB file: {e}", exc_info=True)


def generate_pymol_script(
    predictions_df: pd.DataFrame, # Must contain 'chain_id', 'resid', 'predicted_class'
    config: Dict[str, Any],
    output_pml_path: str,
    pdb_filename: Optional[str] = None # Optional: PDB filename to load in script
):
    """
    Generates a PyMOL (.pml) script to color a structure by flexibility class.

    Args:
        predictions_df: DataFrame with prediction results.
        config: Main configuration dictionary (for colors).
        output_pml_path: Path to save the PyMOL script.
        pdb_filename: Optional name/path of the PDB file to be loaded in the script.
                      If None, assumes the structure is already loaded in PyMOL.
    """
    logger.info(f"Generating PyMOL script: {output_pml_path}")
    colors_map = get_visualization_colors(config)
    class_names_map = get_class_names(config)
    num_classes = config.get('binning', {}).get('num_classes', 5)

    # Validate colors exist for all classes
    if len(colors_map) < num_classes:
        logger.warning(f"Visualization colors defined ({len(colors_map)}) are fewer than number of classes ({num_classes}). Coloring may be incomplete.")

    # Create the script content
    script_lines = [
        f"# PyMOL Script generated by drWiggle to color by flexibility",
        f"# Timestamp: {pd.Timestamp.now()}",
        "bg_color white",
        "set cartoon_fancy_helices, 1",
        "set cartoon_smooth_loops, 1",
        "show cartoon",
        "color grey80, all" # Default color
    ]

    if pdb_filename:
        # Sanitize filename for PyMOL (e.g., escape special characters if necessary)
        safe_pdb_filename = pdb_filename.replace("\\", "/") # Basic path normalization
        script_lines.insert(1, f"load {safe_pdb_filename}") # Load the PDB if specified
        obj_name = os.path.splitext(os.path.basename(safe_pdb_filename))[0]
        script_lines.append(f"disable all") # Disable default representation
        script_lines.append(f"enable {obj_name}")
        script_lines.append(f"show cartoon, {obj_name}")
        script_lines.append(f"color grey80, {obj_name}")


    # Define colors in PyMOL
    pymol_color_names = {} # Map class index to pymol color name
    for class_idx in range(num_classes):
        color_hex = colors_map.get(class_idx) # Colors map should have integer keys
        class_name_safe = class_names_map.get(class_idx, f"class_{class_idx}").replace(" ", "_").replace("-","_").replace("/","_")
        color_name_pymol = f"flex_{class_name_safe}" # Define a PyMOL color name

        if color_hex:
            # Convert hex #RRGGBB to PyMOL [R, G, B] list (0-1 range)
            try:
                color_hex = color_hex.lstrip('#')
                r = int(color_hex[0:2], 16) / 255.0
                g = int(color_hex[2:4], 16) / 255.0
                b = int(color_hex[4:6], 16) / 255.0
                script_lines.append(f"set_color {color_name_pymol}, [{r:.3f}, {g:.3f}, {b:.3f}]")
                pymol_color_names[class_idx] = color_name_pymol
            except (IndexError, ValueError, TypeError):
                logger.warning(f"Invalid hex color format '{color_hex}' for class {class_idx}. Using grey80.")
                pymol_color_names[class_idx] = "grey80" # Fallback color name
        else:
             logger.warning(f"Color not defined for class {class_idx}. Using grey80.")
             pymol_color_names[class_idx] = "grey80" # Fallback color name

    # Color residues based on prediction
    required_cols = ['chain_id', 'resid', 'predicted_class']
    if not all(col in predictions_df.columns for col in required_cols):
         logger.error(f"Predictions DataFrame for PyMOL script must contain columns: {required_cols}. Found: {predictions_df.columns.tolist()}")
         return

    # Ensure 'resid' is integer
    predictions_df['resid'] = predictions_df['resid'].astype(int)

    for class_idx in range(num_classes):
        class_residues = predictions_df[predictions_df['predicted_class'] == class_idx]
        color_name = pymol_color_names.get(class_idx, "grey80")

        if not class_residues.empty:
            selection_parts = []
            # Group by chain to create selections like (chain A and resi 1+5+10)
            for chain, group in class_residues.groupby('chain_id'):
                 res_ids_str = "+".join(map(str, sorted(group['resid'].unique()))) # Sort for cleaner selection string
                 selection_parts.append(f"(chain {chain} and resi {res_ids_str})")

            if selection_parts:
                # Combine selections for the same class with 'or'
                full_selection = " or ".join(selection_parts)
                # Apply coloring to the combined selection
                script_lines.append(f"color {color_name}, ({full_selection})")
            else:
                 logger.debug(f"No residues found for class {class_idx} to color.")

    script_lines.append("zoom vis")
    script_lines.append(f"print('drWiggle coloring applied using colors: {pymol_color_names}')")


    # Write the script to file
    ensure_dir(os.path.dirname(output_pml_path))
    try:
        with open(output_pml_path, 'w') as f:
            f.write("\n".join(script_lines))
        logger.info(f"PyMOL script saved successfully to {output_pml_path}")
    except Exception as e:
        logger.error(f"Failed to write PyMOL script: {e}", exc_info=True)

EOF

echo "Creating utils/visualization.py..."
cat << 'EOF' > drwiggle/utils/visualization.py
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
EOF

echo "Creating temperature/comparison.py..."
cat << 'EOF' > drwiggle/temperature/comparison.py
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
EOF

echo "Creating pipeline.py..."
cat << 'EOF' > drwiggle/pipeline.py
import logging
import os
import time
import traceback
from typing import Dict, Any, Optional, List, Union, Tuple
import sys

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
from drwiggle.models import get_model_instance, get_enabled_models, BaseClassifier
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
            # Find saved models in the directory
            found_files = glob.glob(os.path.join(models_dir, "*.joblib"))
            # Exclude binner file
            model_files = [f for f in found_files if not os.path.basename(f).startswith('binner')]
            if not model_files:
                 logger.error(f"No models specified and no '.joblib' model files found in {models_dir}. Cannot evaluate.")
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
                model_path = os.path.join(models_dir, f'{model_name}.joblib')
                if os.path.exists(model_path):
                    try:
                        # Pass current config context for loading
                        model = BaseClassifier.load(model_path, config=self.config)
                        self.models[model_name] = model # Store loaded model
                        # Store feature names from loaded model if pipeline doesn't have them yet
                        if self.feature_names_in_ is None and model.feature_names_in_:
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

        model_path = os.path.join(models_dir, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            logger.error(f"Model file for '{model_name}' not found at {model_path}. Cannot predict.")
            return None

        try:
            # Check if model already loaded in pipeline instance
            model = self.models.get(model_name)
            if not model:
                 model = BaseClassifier.load(model_path, config=self.config)
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
                 raise ValueError(f"Input data missing required features for model '{model_name}': {missing_features}")

            X_pred, _, _ = prepare_data_for_model(df_input, self.config, target_col=None, features=self.feature_names_in_)

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
        # Include original identifiers if possible
        id_cols = [col for col in ['domain_id', 'chain_id', 'resid', 'icode', 'resname'] if col in df_input.columns]
        result_df = df_input[id_cols].reset_index(drop=True).copy()
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

EOF

echo "Creating cli.py..."
cat << 'EOF' > drwiggle/cli.py
import logging
import click
import os
import sys
import pandas as pd
from typing import Optional, Tuple

# --- Configure Logging ---
# Basic config here, gets potentially overridden by config file later in load_config
logging.basicConfig(
    level=logging.INFO, # Default level
    format='%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)] # Log to stdout
)
# Silence overly verbose libraries by default
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("numexpr").setLevel(logging.WARNING)

logger = logging.getLogger("drwiggle.cli") # Logger specific to CLI

# --- Import Core Components ---
# Defer heavy imports until commands are run if possible
# (though pipeline import might trigger others)
from drwiggle.config import load_config
from drwiggle.pipeline import Pipeline


# --- Helper Functions ---
def _setup_pipeline(ctx, config_path: Optional[str], param_overrides: Optional[Tuple[str]], cli_option_overrides: dict) -> Pipeline:
    """Loads config and initializes the pipeline."""
    try:
        # Pass CLI overrides directly to load_config
        # Resolve paths relative to the current working directory (where CLI is run)
        cfg = load_config(
            config_path=config_path,
            cli_overrides=cli_option_overrides,
            param_overrides=param_overrides,
            resolve_paths_base_dir=os.getcwd() # Resolve relative to CWD
        )
        # Store config in context for potential use by other commands if needed
        ctx.obj = cfg
        pipeline = Pipeline(cfg)
        return pipeline
    except FileNotFoundError as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)
    except (ValueError, TypeError, KeyError) as e:
         logger.error(f"Configuration Error: Invalid setting or structure - {e}", exc_info=True)
         sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
        sys.exit(1)

# --- Click CLI Definition ---

@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.version_option(version="1.0.0", package_name="drwiggle") # Assumes setup.py version
@click.option('--config', '-c', type=click.Path(exists=True, dir_okay=False), help='Path to custom YAML config file.')
@click.option('--param', '-p', multiple=True, help='Override config param (key.subkey=value). Can be used multiple times.')
@click.pass_context # Pass context to store config
def cli(ctx, config, param):
    """
    drWiggle: Protein Flexibility Classification Framework.

    Train models, evaluate performance, predict flexibility, and analyze results
    across different temperatures based on RMSF data and structural features.

    Configuration is loaded from default_config.yaml, overridden by the --config file,
    environment variables (DRWIGGLE_*), and finally --param options.
    """
    # Store base config path and params in context for commands to use
    # The actual config loading happens within each command using _setup_pipeline
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['param_overrides'] = param
    logger.info("drWiggle CLI started.")


@cli.command()
@click.option("--model", '-m', help="Model(s) to train (comma-separated, e.g., 'random_forest,neural_network'). Default: all enabled in config.")
@click.option("--input", '-i', type=click.Path(resolve_path=True), help="Input data file/pattern. Overrides 'dataset.file_pattern' in config.")
@click.option("--temperature", '-t', type=str, help="Temperature context (e.g., 320). Overrides 'temperature.current'. REQUIRED if data pattern uses {temperature}.")
@click.option("--binning", '-b', type=click.Choice(["kmeans", "quantile"], case_sensitive=False), help="Override binning method.")
@click.option("--output-dir", '-o', type=click.Path(resolve_path=True), help="Override 'paths.output_dir'.")
@click.option("--models-dir", type=click.Path(resolve_path=True), help="Override 'paths.models_dir'.")
@click.pass_context
def train(ctx, model, input, temperature, binning, output_dir, models_dir):
    """Train flexibility classification model(s)."""
    logger.info("=== Train Command Initiated ===")
    # Prepare CLI overrides dictionary for load_config
    cli_overrides = {}
    if temperature: cli_overrides.setdefault('temperature', {})['current'] = temperature
    if binning: cli_overrides.setdefault('binning', {})['method'] = binning
    if output_dir: cli_overrides.setdefault('paths', {})['output_dir'] = output_dir
    if models_dir: cli_overrides.setdefault('paths', {})['models_dir'] = models_dir
    # Input override needs careful handling - pass directly to pipeline method
    # if input: cli_overrides.setdefault('dataset', {})['file_pattern'] = input # This isn't quite right, input can be path

    pipeline = _setup_pipeline(ctx, ctx.obj['config_path'], ctx.obj['param_overrides'], cli_overrides)

    # Temperature check: crucial if file_pattern uses {temperature} and --input not given
    if input is None and '{temperature}' in pipeline.config['dataset']['file_pattern']:
         current_temp = pipeline.config.get("temperature", {}).get("current")
         if current_temp is None:
              logger.error("Training data pattern requires {temperature}, but temperature not set via --temperature or config.")
              sys.exit(1)
         logger.info(f"Using temperature {current_temp} for finding training data.")


    model_list = model.split(',') if model else None # Pass None to train all enabled

    try:
        pipeline.train(model_names=model_list, data_path=input)
        logger.info("=== Train Command Finished Successfully ===")
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


@cli.command()
@click.option("--model", '-m', help="Model(s) to evaluate (comma-separated). Default: All models found in models_dir.")
@click.option("--input", '-i', type=click.Path(resolve_path=True), help="Evaluate on specific data file/pattern. Default: Use test split from training data source.")
@click.option("--temperature", '-t', type=str, help="Temperature context for loading models/data (e.g., 320). REQUIRED if default data pattern needs it.")
@click.option("--output-dir", '-o', type=click.Path(resolve_path=True), help="Override 'paths.output_dir'.")
@click.option("--models-dir", type=click.Path(resolve_path=True), help="Override 'paths.models_dir'.")
@click.pass_context
def evaluate(ctx, model, input, temperature, output_dir, models_dir):
    """Evaluate trained classification model(s)."""
    logger.info("=== Evaluate Command Initiated ===")
    cli_overrides = {}
    if temperature: cli_overrides.setdefault('temperature', {})['current'] = temperature
    if output_dir: cli_overrides.setdefault('paths', {})['output_dir'] = output_dir
    if models_dir: cli_overrides.setdefault('paths', {})['models_dir'] = models_dir

    pipeline = _setup_pipeline(ctx, ctx.obj['config_path'], ctx.obj['param_overrides'], cli_overrides)

    # Temperature check if default test set derivation needs it
    if input is None and '{temperature}' in pipeline.config['dataset']['file_pattern']:
         current_temp = pipeline.config.get("temperature", {}).get("current")
         if current_temp is None:
              logger.error("Deriving test set requires {temperature} in data pattern, but temperature not set via --temperature or config.")
              sys.exit(1)
         logger.info(f"Using temperature {current_temp} for deriving test set.")

    model_list = model.split(',') if model else None

    try:
        pipeline.evaluate(model_names=model_list, data_path=input)
        logger.info("=== Evaluate Command Finished Successfully ===")
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


@cli.command()
@click.option("--input", '-i', type=click.Path(exists=True, dir_okay=False, resolve_path=True), required=True, help="Input data file (CSV) for prediction.")
@click.option("--model", '-m', type=str, help="Model name to use. Default: 'random_forest'.")
@click.option("--output", '-o', type=click.Path(resolve_path=True), help="Output file path for predictions (CSV). Default: derive from input filename.")
@click.option("--temperature", '-t', type=str, help="Temperature context for loading model (e.g., 320). Sets 'temperature.current' in config.")
@click.option("--probabilities", is_flag=True, default=False, help="Include class probabilities in output.")
@click.option("--models-dir", type=click.Path(resolve_path=True), help="Override 'paths.models_dir'.")
@click.pass_context
def predict(ctx, input, model, output, temperature, probabilities, models_dir):
    """Predict flexibility classes for new data."""
    logger.info("=== Predict Command Initiated ===")
    cli_overrides = {}
    if temperature: cli_overrides.setdefault('temperature', {})['current'] = temperature
    if models_dir: cli_overrides.setdefault('paths', {})['models_dir'] = models_dir
    # Store probability flag for pipeline to access
    cli_overrides.setdefault('cli_options', {})['predict_probabilities'] = probabilities

    pipeline = _setup_pipeline(ctx, ctx.obj['config_path'], ctx.obj['param_overrides'], cli_overrides)

    if not output:
        base, ext = os.path.splitext(input)
        output = f"{base}_predictions.csv"
        logger.info(f"Output path not specified, defaulting to: {output}")

    try:
        predictions_df = pipeline.predict(data=input, model_name=model, output_path=output)
        # Predict method handles saving if output_path is given
        if predictions_df is not None:
             # This happens if output_path wasn't specified or saving failed
             logger.info("Prediction method returned DataFrame (likely because output_path was None or saving failed).")
        logger.info("=== Predict Command Finished Successfully ===")
    except Exception as e:
        logger.error(f"Prediction pipeline failed: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


@cli.command()
@click.option("--pdb", required=True, help="PDB ID (e.g., '1AKE') or path to a local PDB file.")
@click.option("--model", '-m', type=str, help="Model name to use for prediction. Default: 'random_forest'.")
@click.option("--temperature", '-t', type=str, help="Temperature context for prediction model (e.g., 320). Sets 'temperature.current'.")
@click.option("--output-prefix", '-o', type=click.Path(resolve_path=True), help="Output prefix for generated files (e.g., ./output/1ake_flex). Default: '{output_dir}/pdb_vis/{pdb_id}_{model}_flexibility'")
@click.option("--models-dir", type=click.Path(resolve_path=True), help="Override 'paths.models_dir'.")
@click.option("--pdb-cache-dir", type=click.Path(resolve_path=True), help="Override 'paths.pdb_cache_dir'.")
@click.pass_context
def process_pdb(ctx, pdb, model, temperature, output_prefix, models_dir, pdb_cache_dir):
    """Fetch/Parse PDB, Extract Features, Predict Flexibility, and Generate Visualizations."""
    logger.info("=== Process PDB Command Initiated ===")
    cli_overrides = {}
    if temperature: cli_overrides.setdefault('temperature', {})['current'] = temperature
    if models_dir: cli_overrides.setdefault('paths', {})['models_dir'] = models_dir
    if pdb_cache_dir: cli_overrides.setdefault('paths', {})['pdb_cache_dir'] = pdb_cache_dir
    # Ensure PDB processing is enabled in the loaded config
    cli_overrides.setdefault('pdb', {})['enabled'] = True

    pipeline = _setup_pipeline(ctx, ctx.obj['config_path'], ctx.obj['param_overrides'], cli_overrides)

    try:
        pipeline.process_pdb(pdb_input=pdb, model_name=model, output_prefix=output_prefix)
        logger.info("=== Process PDB Command Finished Successfully ===")
    except Exception as e:
        logger.error(f"PDB processing pipeline failed: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


@cli.command()
@click.option("--input", '-i', type=click.Path(exists=True, dir_okay=False, resolve_path=True), required=True, help="Input RMSF data file (CSV) for analysis.")
@click.option("--temperature", '-t', type=str, help="Temperature context (e.g., 320).")
@click.option("--output-dir", '-o', type=click.Path(resolve_path=True), help="Directory to save the plot. Overrides 'paths.output_dir'.")
@click.option("--models-dir", type=click.Path(resolve_path=True), help="Directory containing saved binner. Overrides 'paths.models_dir'.")
@click.pass_context
def analyze_distribution(ctx, input, temperature, output_dir, models_dir):
    """Analyze RMSF distribution and visualize binning boundaries."""
    logger.info("=== Analyze Distribution Command Initiated ===")
    cli_overrides = {}
    if temperature: cli_overrides.setdefault('temperature', {})['current'] = temperature
    if output_dir: cli_overrides.setdefault('paths', {})['output_dir'] = output_dir
    if models_dir: cli_overrides.setdefault('paths', {})['models_dir'] = models_dir

    pipeline = _setup_pipeline(ctx, ctx.obj['config_path'], ctx.obj['param_overrides'], cli_overrides)

    plot_filename = f"rmsf_distribution_analysis_{temperature or 'default'}.png"
    plot_path = os.path.join(pipeline.config['paths']['output_dir'], plot_filename)

    try:
        pipeline.analyze_rmsf_distribution(input_data_path=input, output_plot_path=plot_path)
        logger.info("=== Analyze Distribution Command Finished Successfully ===")
    except Exception as e:
        logger.error(f"RMSF distribution analysis failed: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


@cli.command()
@click.option("--model", '-m', type=str, help="Model name to focus comparison on (optional).")
@click.option("--output-dir", '-o', type=click.Path(resolve_path=True), help="Override base 'paths.output_dir' for finding results and saving comparison.")
@click.pass_context
def compare_temperatures(ctx, model, output_dir):
    """Compare classification results across different temperatures."""
    logger.info("=== Compare Temperatures Command Initiated ===")
    cli_overrides = {}
    # Output dir override applies to the base dir where temp results are sought
    if output_dir: cli_overrides.setdefault('paths', {})['output_dir'] = output_dir
    # Temperature override doesn't make sense here as we compare multiple temps

    pipeline = _setup_pipeline(ctx, ctx.obj['config_path'], ctx.obj['param_overrides'], cli_overrides)

    try:
        pipeline.run_temperature_comparison(model_name=model)
        logger.info("=== Compare Temperatures Command Finished Successfully ===")
    except Exception as e:
        logger.error(f"Temperature comparison failed: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


@cli.command()
@click.option("--predictions", type=click.Path(exists=True, dir_okay=False, resolve_path=True), required=True, help="Path to the predictions CSV file (must contain 'predicted_class').")
@click.option("--output-dir", '-o', type=click.Path(resolve_path=True), help="Directory to save visualizations. Overrides 'paths.output_dir'.")
@click.pass_context
def visualize(ctx, predictions, output_dir):
    """Generate visualizations from saved prediction files."""
    logger.info("=== Visualize Command Initiated ===")
    cli_overrides = {}
    if output_dir: cli_overrides.setdefault('paths', {})['output_dir'] = output_dir
    # Temperature override might be needed if config depends on it for vis settings
    # Add --temperature option if necessary later.

    pipeline = _setup_pipeline(ctx, ctx.obj['config_path'], ctx.obj['param_overrides'], cli_overrides)

    try:
        pipeline.visualize_results(predictions_path=predictions, output_dir=output_dir) # Pass specified output dir
        logger.info("=== Visualize Command Finished Successfully ===")
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


# Main entry point for script execution
if __name__ == '__main__':
    # Set process title if possible (useful for monitoring)
    try:
        import setproctitle
        setproctitle.setproctitle("drwiggle_cli")
    except ImportError:
        pass
    # Execute the Click application
    cli()
EOF

echo ""
echo "--- drWiggle Engine Generation: Script 2 Complete ---"
echo "Model implementations, utilities, pipeline, and CLI created."
echo "Project structure in 'drwiggle_project' should now be complete."
echo "Next steps:"
echo "1. Place your input data CSV files into the 'drwiggle_project/data/' directory."
echo "2. Navigate to 'drwiggle_project/drwiggle/'."
echo "3. Create and activate a virtual environment (e.g., python -m venv venv && source venv/bin/activate)."
echo "4. Install the package: pip install -e ."
echo "5. Run commands from the 'drwiggle_project/' directory, e.g., 'drwiggle train -t 320'"
echo "6. Review and potentially adjust paths in 'drwiggle/drwiggle/default_config.yaml' if you run commands from a different location."
