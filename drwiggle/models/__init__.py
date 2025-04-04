# --- File: drwiggle/models/__init__.py ---
import logging
from typing import Dict, Any, Type, Optional, List

from .base import BaseClassifier
from .random_forest import RandomForestClassifier
from .neural_network import NeuralNetworkClassifier
from .xgboost import XGBoostClassifier 
from .lightgbm import LightGBMClassifier

logger = logging.getLogger(__name__)

# Registry of model names to classes
MODEL_REGISTRY: Dict[str, Type[BaseClassifier]] = {
    "lightgbm": LightGBMClassifier,
    "xgboost": XGBoostClassifier, 
    "random_forest": RandomForestClassifier,
    "neural_network": NeuralNetworkClassifier,
}

def get_model_instance(config: Dict[str, Any], model_name: str) -> BaseClassifier:
    """
    Factory function to get an instance of a registered model class.

    Args:
        config: The main configuration dictionary.
        model_name: The name of the model to instantiate.

    Returns:
        An instance of the requested model class.

    Raises:
        ValueError: If the model_name is not registered.
    """
    model_cls = MODEL_REGISTRY.get(model_name.lower()) # Use lower case for lookup
    if model_cls:
        logger.info(f"Instantiating model: '{model_name}'")
        return model_cls(config=config, model_name=model_name)
    else:
        raise ValueError(f"Unknown model name: '{model_name}'. Registered models: {list(MODEL_REGISTRY.keys())}")

def get_model_class(config: Dict[str, Any], model_name: str) -> Optional[Type[BaseClassifier]]:
    """
    Factory function to get the class type of a registered model.

    Args:
        config: The main configuration dictionary (currently unused but kept for signature).
        model_name: The name of the model class to retrieve.

    Returns:
        The class type corresponding to the model name, or None if not found.
    """
    model_cls = MODEL_REGISTRY.get(model_name.lower())
    if not model_cls:
        logger.warning(f"Model class not found in registry for name: '{model_name}'")
    return model_cls


def get_enabled_models(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Gets a dictionary of model configurations for models marked as enabled.

    Args:
        config: The main configuration dictionary.

    Returns:
        A dictionary where keys are enabled model names and values are their specific configs.
    """
    models_config = config.get("models", {})
    enabled_models = {}
    for model_name, model_cfg in models_config.items():
        # Check if the item is a dictionary (model config) and has 'enabled: true'
        if isinstance(model_cfg, dict) and model_cfg.get("enabled", False):
            # Check if the model name is actually registered
            if model_name in MODEL_REGISTRY:
                 enabled_models[model_name] = model_cfg
            else:
                 # Allow 'common' section, but warn about other unknown enabled sections
                 if model_name != 'common':
                      logger.warning(f"Model '{model_name}' is enabled in config but not found in MODEL_REGISTRY. Ignoring.")

    return enabled_models

# --- End File: drwiggle/models/__init__.py ---