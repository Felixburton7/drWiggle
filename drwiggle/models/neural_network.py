import logging
import os
import time
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd
import joblib # Add this import
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
# Use helpers for joblib saving/loading within the class methods now
from drwiggle.utils.helpers import progress_bar, save_object, load_object, ensure_dir

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

    # <<<--- START MODIFIED SAVE METHOD --- >>>
    def save(self, path: str):
        """Save the model state_dict using torch.save and the rest using joblib."""
        # Derive paths for state_dict (.pt) and metadata (.joblib)
        base_path, _ = os.path.splitext(path)
        state_dict_path = f"{base_path}_statedict.pt"
        # Ensure meta_path IS the original path passed (expected to be .joblib)
        meta_path = path # Assume path is e.g. .../neural_network.joblib

        # Check fitted status (from BaseClassifier.save logic)
        if not self._fitted:
            raise RuntimeError(f"Cannot save model '{self.model_name}' because it has not been fitted.")
        try: # Add basic error handling around ensure_dir too
            ensure_dir(os.path.dirname(path)) # Ensure directory for meta_path exists
        except OSError as e:
             logger.error(f"Failed to create directory for saving model at {path}: {e}")
             raise IOError(f"Could not create directory for model file: {path}") from e

        # 1. Save Model State Dict using torch.save
        if self.model is None:
            logger.warning(f"Model object is None for '{self.model_name}'. Cannot save model state dict.")
        else:
            try:
                ensure_dir(os.path.dirname(state_dict_path)) # Ensure dir exists
                torch.save(self.model.state_dict(), state_dict_path)
                logger.info(f"NeuralNetwork state_dict saved to {state_dict_path}")
            except Exception as e:
                logger.error(f"Failed to save NeuralNetwork state_dict to {state_dict_path}: {e}", exc_info=True)
                raise IOError(f"Could not save NN state_dict to {state_dict_path}") from e

        # 2. Save Metadata (scaler, config, etc.) using joblib
        if not hasattr(self, 'scaler') or not hasattr(self.scaler, 'mean_'):
             logger.warning(f"Scaler not found or not fitted for model '{self.model_name}'. Saving metadata without scaler state.")
             scaler_state = None
        else:
             scaler_state = self.scaler

        meta_state = {
            # DO NOT save model_state_dict here
            'scaler_state': scaler_state,
            'feature_names_in_': self.feature_names_in_,
            'config': self.config,
            'model_config': self.model_config,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'history': self.history,
            'fitted': self._fitted # Still save fitted status based on training completion
        }
        try:
            save_object(meta_state, meta_path) # Use helper which uses joblib
            logger.info(f"NeuralNetwork metadata saved to {meta_path}")
        except Exception as e:
            logger.error(f"Failed to save NeuralNetwork metadata to {meta_path}: {e}", exc_info=True)
            # Should we delete the state_dict if meta fails? Maybe.
            if self.model is not None and os.path.exists(state_dict_path):
                 try: os.remove(state_dict_path)
                 except OSError: pass
            raise IOError(f"Could not save NN metadata state to {meta_path}") from e
    # <<<--- END MODIFIED SAVE METHOD --- >>>

    # <<<--- START MODIFIED LOAD METHOD --- >>>
    @classmethod
    def load(cls, path: str, config: Optional[Dict[str, Any]] = None) -> 'NeuralNetworkClassifier':
        """Load metadata using joblib and model state_dict using torch.load."""
        # --- NEW Simpler Path Derivation ---
        meta_path = path # Assume path is the .joblib file
        base_path, _ = os.path.splitext(meta_path)
        state_dict_path = f"{base_path}_statedict.pt" # Derive state_dict path from joblib path
        # --- End NEW Path Derivation ---

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")
        if not os.path.exists(state_dict_path):
             raise FileNotFoundError(f"Model state_dict file not found at {state_dict_path}")

        logger.info(f"Loading model '{cls.__name__}' metadata from {meta_path} and state_dict from {state_dict_path}...")

        # 1. Load Metadata using joblib (via helper)
        try:
            state = load_object(meta_path)
        except Exception as e:
            logger.error(f"Failed to load metadata from {meta_path}: {e}", exc_info=True)
            raise IOError(f"Could not load NN metadata from {meta_path}") from e

        # Basic validation of loaded metadata state
        required_keys = ['scaler_state', 'feature_names_in_', 'model_config', 'model_name', 'num_classes', 'fitted']
        if not all(key in state for key in required_keys):
             raise ValueError(f"Loaded NN metadata from {meta_path} is missing required keys. Found: {list(state.keys())}")

        # Use loaded config if available, otherwise fall back to provided runtime config
        load_config = state.get('config', config)
        if load_config is None:
             raise ValueError("Cannot load NN model: No configuration found in saved state or provided at runtime.")

        # Re-instantiate the class
        instance = cls(config=load_config, model_name=state['model_name'])

        # Restore state from metadata
        instance.scaler = state['scaler_state']
        instance.feature_names_in_ = state['feature_names_in_']
        instance.num_classes = state['num_classes']
        instance.history = state.get('history', {'train_loss': [], 'val_loss': [], 'val_accuracy': []}) # Handle older saves
        instance._fitted = state['fitted']

        # --- Determine device for loading state_dict ---
        runtime_config_system = load_config.get("system", {}) # Use potentially loaded config
        gpu_enabled_cfg = runtime_config_system.get("gpu_enabled", "auto")
        force_cpu = gpu_enabled_cfg == False
        if not force_cpu and torch.cuda.is_available(): device = torch.device("cuda")
        elif not force_cpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = torch.device("mps")
        else: device = torch.device("cpu")
        instance.device = device
        logger.info(f"Loading NeuralNetwork state_dict to device {device}")

        # 2. Re-create model architecture & Load State Dict using torch.load
        input_dim = len(instance.feature_names_in_) if instance.feature_names_in_ else None
        if input_dim is None and instance.scaler and hasattr(instance.scaler, 'n_features_in_'):
              input_dim = instance.scaler.n_features_in_
        if input_dim is None:
              raise ValueError("Cannot determine input dimension for loading NN model (missing feature names and scaler info).")

        try:
            instance.model = DrWiggleNN(input_dim, instance.num_classes, instance.model_config).to(instance.device)
            # Load state dict - use weights_only=False explicitly if needed for compatibility or trust source
            # Using weights_only=True (default in PyTorch >= 2.6) is safer if possible
            try:
                # Try default first (weights_only=True in newer PyTorch)
                 model_state_dict = torch.load(state_dict_path, map_location=device)
            except (TypeError, RuntimeError, AttributeError, EOFError) as e:
                 # Fallback for older PyTorch versions or if weights_only=True fails unexpectedly
                 # Check the error message - if it explicitly mentions weights_only, try False
                 if 'weights_only' in str(e):
                      logger.warning(f"torch.load with weights_only=True failed. Attempting with weights_only=False. Ensure the file source '{state_dict_path}' is trusted.")
                      model_state_dict = torch.load(state_dict_path, map_location=device, weights_only=False)
                 else:
                      raise # Re-raise other torch load errors

            instance.model.load_state_dict(model_state_dict)
            instance.model.eval() # Set to evaluation mode
            logger.info("Model architecture recreated and state dict loaded successfully.")
        except FileNotFoundError:
             logger.error(f"Model state_dict file not found: {state_dict_path}")
             raise
        except Exception as e:
            logger.error(f"Failed to load torch state_dict from {state_dict_path}: {e}", exc_info=True)
            # Mark as not fitted if state dict loading fails
            instance.model = None
            instance._fitted = False
            raise IOError(f"Could not load NN state_dict from {state_dict_path}") from e

        # Validate scaler (moved check here)
        if instance.scaler and not hasattr(instance.scaler, 'mean_'):
             logger.warning(f"Loaded scaler for model '{instance.model_name}' appears not to be fitted.")
             # Mark as not fitted if scaler missing/unfitted?
             # instance._fitted = False # Let's assume fitted status from meta_state is primary indicator

        if not instance._fitted:
             logger.warning(f"Loaded model '{instance.model_name}' from {meta_path} is marked as not fitted (check metadata file).")

        logger.info(f"Model '{instance.model_name}' loaded successfully.")
        return instance
    # <<<--- END MODIFIED LOAD METHOD --- >>>


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
                          for batch_X_val, batch_y_val in val_loader_trial: # Indent Level 2
                               batch_X_val, batch_y_val = batch_X_val.to(self.device), batch_y_val.to(self.device)
                               outputs_val = temp_model(batch_X_val)
                               if isinstance(temp_criterion, nn.MSELoss): # Indent Level 3
                                    loss_val = temp_criterion(outputs_val, batch_y_val.unsqueeze(-1))
                                    preds = torch.round(outputs_val.squeeze()).clamp(0, self.num_classes - 1).long()
                               else: # Indent Level 3
                                    loss_val = temp_criterion(outputs_val, batch_y_val)
                                    _, preds = torch.max(outputs_val, 1)
                               # Correct indentation for these lines:
                               epoch_val_loss += loss_val.item() * batch_X_val.size(0) # <-- Must be Level 3
                               correct_preds += (preds == batch_y_val).sum().item() # <-- Must be Level 3
                               total_preds += batch_y_val.size(0) # <-- Must be Level 3

                     # This part should be at Indent Level 2
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