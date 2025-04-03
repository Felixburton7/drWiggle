import logging
from typing import Dict, Type, Any

from .base import BaseClassifier
from .random_forest import RandomForestClassifier
from .neural_network import NeuralNetworkClassifier
from drwiggle.config import get_model_config # Import needed config helper

logger = logging.getLogger(__name__)

# Model Registry
_MODEL_REGISTRY: Dict[str, Type[BaseClassifier]] = {
    "random_forest": RandomForestClassifier,
    "neural_network": NeuralNetworkClassifier,
    # Add other models here as they are implemented
}

def get_model_class(model_name: str) -> Type[BaseClassifier]:
    """
    Factory function to get a model class based on its name.

    Args:
        model_name: The name of the model (e.g., "random_forest").

    Returns:
        The corresponding model class (subclass of BaseClassifier).

    Raises:
        ValueError: If the model_name is not found in the registry.
    """
    model_name_lower = model_name.lower()
    model_cls = _MODEL_REGISTRY.get(model_name_lower)
    if model_cls:
        return model_cls
    else:
        raise ValueError(f"Unknown model name: '{model_name}'. Available models: {list(_MODEL_REGISTRY.keys())}")

def get_model_instance(config: Dict[str, Any], model_name: str) -> BaseClassifier:
    """
    Factory function to get an initialized model instance based on config and name.

    Args:
        config: The main configuration dictionary.
        model_name: The name of the model.

    Returns:
        An initialized instance of the specified model class.

    Raises:
        ValueError: If the model_name is unknown or the class cannot be instantiated.
    """
    model_cls = get_model_class(model_name)
    logger.info(f"Instantiating model: '{model_name}'")
    # The model's __init__ should handle extracting its specific config using get_model_config
    try:
        instance = model_cls(config=config, model_name=model_name)
        return instance
    except Exception as e:
        logger.error(f"Failed to instantiate model '{model_name}': {e}", exc_info=True)
        raise ValueError(f"Could not instantiate model '{model_name}'.")


def get_enabled_models(config: Dict[str, Any]) -> Dict[str, Type[BaseClassifier]]:
    """
    Gets a dictionary of enabled model names and their corresponding classes.

    Args:
        config: The main configuration dictionary.

    Returns:
        A dictionary {model_name: ModelClass} for enabled models.
    """
    enabled_models = {}
    models_config = config.get("models", {})
    for name, model_cls in _MODEL_REGISTRY.items():
        model_cfg = get_model_config(config, name) # Use helper to get merged config
        if model_cfg.get("enabled", False):
            enabled_models[name] = model_cls
            logger.debug(f"Model '{name}' is enabled.")
        else:
             logger.debug(f"Model '{name}' is disabled.")
    return enabled_models

