import os
import logging
import yaml
from typing import Dict, Any, Optional, List, Union, Tuple
import copy # For deep copying config
import warnings

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILENAME = "default_config.yaml"

def deep_merge(base_dict: Dict, overlay_dict: Dict) -> Dict:
    """
    Recursively merge two dictionaries. Values from overlay_dict take precedence.
    Modifies base_dict in place. Handles nested dictionaries.
    """
    for key, value in overlay_dict.items():
        if key in base_dict:
            # If both are dicts, recurse
            if isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_merge(base_dict[key], value)
            # If overlay value is not None, overwrite (allows explicit nulls)
            elif value is not None:
                 base_dict[key] = value
            # If overlay value is None, keep base value (allows unsetting via None in overlay)
            # This behavior might need adjustment depending on desired merge logic for None.
            # Currently: None in overlay means "don't change the base value".
            # To explicitly set a value to None, the overlay should contain `key: null` in YAML.
            else:
                pass # Keep base_dict[key]
        else:
            # Key not in base, add it
            base_dict[key] = value
    return base_dict


def _get_default_config_path() -> str:
    """Find the default config file path relative to this config.py script."""
    # This assumes default_config.yaml is in the same directory as config.py
    return os.path.join(os.path.dirname(__file__), DEFAULT_CONFIG_FILENAME)

def _parse_value(value: str) -> Any:
    """Try to parse string value into bool, int, float, list, or keep as string."""
    val_lower = value.lower()
    if val_lower == "true": return True
    if val_lower == "false": return False
    if val_lower in ["null", "none", ""]: return None
    try: return int(value)
    except ValueError:
        try: return float(value)
        except ValueError:
            # Try parsing comma-separated lists (e.g., "1,2,3" -> [1, 2, 3])
            if ',' in value:
                try:
                    # Attempt to parse each element as int/float/string
                    return [_parse_value(item.strip()) for item in value.split(',')]
                except Exception:
                    pass # Fallback to string if list parsing fails
            return value # Keep as string

def _set_nested_value(d: Dict, keys: List[str], value: Any):
    """Set a value in a nested dictionary using a list of keys."""
    for key in keys[:-1]:
        # If the key exists but isn't a dict, behavior is undefined.
        # For robustness, maybe raise error or overwrite? Currently overwrites.
        if key not in d or not isinstance(d[key], dict):
             d[key] = {} # Ensure intermediate keys are dicts
        d = d[key]

    # Check if the final key exists and is a dictionary, and the value is also a dictionary
    # This indicates a potential merge scenario instead of overwrite, especially for HPO params
    if keys[-1] in d and isinstance(d[keys[-1]], dict) and isinstance(value, dict):
        deep_merge(d[keys[-1]], value) # Merge if both are dicts
    else:
        d[keys[-1]] = value # Otherwise, overwrite/set

def _get_env_var_config() -> Dict[str, Any]:
    """Load configuration overrides from environment variables (prefixed DRWIGGLE_)."""
    config = {}
    prefix = "DRWIGGLE_"
    for env_var, value_str in os.environ.items():
        if env_var.startswith(prefix):
            key_str = env_var[len(prefix):].lower()
            # Split by double underscore for nesting, single for parts of name
            keys = key_str.split('__') # e.g., DRWIGGLE_MODELS__RANDOM_FOREST__N_ESTIMATORS
            parsed_keys = []
            for k in keys:
                 parsed_keys.extend(k.split('_')) # Split remaining parts by single underscore

            value = _parse_value(value_str)
            try:
                 _set_nested_value(config, parsed_keys, value)
                 logger.debug(f"Loaded from env: {env_var} -> {parsed_keys} = {value}")
            except (TypeError, IndexError) as e:
                 logger.warning(f"Could not set nested value from env var {env_var} (keys: {parsed_keys}): {e}")
    return config

def _parse_param_overrides(params: Optional[Tuple[str]]) -> Dict[str, Any]:
    """Parse --param key.subkey=value overrides from CLI."""
    config = {}
    if not params:
        return config
    for param in params:
        if '=' not in param:
            logger.warning(f"Ignoring invalid param override (no '='): {param}")
            continue
        key_str, value_str = param.split('=', 1)
        keys = key_str.split('.') # Use dot for nesting in CLI params
        value = _parse_value(value_str)
        try:
            _set_nested_value(config, keys, value)
            logger.debug(f"Loaded from CLI param: {param} -> {keys} = {value}")
        except (TypeError, IndexError) as e:
             logger.warning(f"Could not set nested value from CLI param {param} (keys: {keys}): {e}")
    return config

def _resolve_paths(config: Dict[str, Any], base_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolve relative paths in the config, making them absolute.
    Paths under the 'paths' key ending in '_dir' or '_file' are resolved.
    Relative paths are resolved relative to 'base_dir'.
    If 'base_dir' is None, uses the current working directory.
    """
    resolved_config = copy.deepcopy(config)
    if base_dir is None:
        base_dir = os.getcwd()
        logger.debug(f"No base directory provided for path resolution, using CWD: {base_dir}")

    paths_config = resolved_config.get('paths', {})
    if paths_config:
        for key, value in paths_config.items():
            if isinstance(value, str) and (key.endswith('_dir') or key.endswith('_file') or key.endswith('_path')):
                if not os.path.isabs(value):
                    abs_path = os.path.abspath(os.path.join(base_dir, value))
                    paths_config[key] = abs_path
                    logger.debug(f"Resolved relative path '{key}': '{value}' -> '{abs_path}' (relative to {base_dir})")
                else:
                     logger.debug(f"Path '{key}' is already absolute: '{value}'")

    # Resolve dssp_path if present and relative
    pdb_config = resolved_config.get('pdb', {})
    if pdb_config:
         dssp_path = pdb_config.get('dssp_path')
         if dssp_path and isinstance(dssp_path, str) and not os.path.isabs(dssp_path):
             # Assume dssp_path is relative to base_dir if not absolute
             # Although usually it refers to an executable in PATH or an absolute location
             abs_dssp_path = os.path.abspath(os.path.join(base_dir, dssp_path))
             pdb_config['dssp_path'] = abs_dssp_path
             logger.debug(f"Resolved relative dssp_path: '{dssp_path}' -> '{abs_dssp_path}'")

    return resolved_config


def _template_config(config: Dict[str, Any], temperature: Union[int, str]) -> Dict[str, Any]:
    """Recursively replace {temperature} placeholders in config strings."""
    templated_config = copy.deepcopy(config) # Avoid modifying original

    def _recursive_replace(item):
        if isinstance(item, dict):
            return {k: _recursive_replace(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [_recursive_replace(elem) for elem in item]
        elif isinstance(item, str):
            # Format the string using a dict to avoid errors if placeholder missing
            try:
                 return item.format(temperature=temperature)
            except KeyError:
                 return item # Return original string if placeholder not found
        else:
            return item

    return _recursive_replace(templated_config)


def load_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    param_overrides: Optional[Tuple[str]] = None,
    resolve_paths_base_dir: Optional[str] = None # Base dir for resolving relative paths
) -> Dict[str, Any]:
    """
    Loads and merges configuration from multiple sources with precedence.

    Precedence order (highest first):
    1. Command-line --param overrides (`param_overrides`)
    2. Command-line specific options (`cli_overrides`, e.g., --temperature, --output-dir)
    3. Environment variables (DRWIGGLE_*)
    4. User-provided config file (`config_path`)
    5. Default config file (`drwiggle/default_config.yaml`)

    Args:
        config_path: Path to a user-specific YAML config file.
        cli_overrides: Dict of overrides from specific CLI options (e.g., {'temperature': {'current': 320}}).
        param_overrides: Tuple of "key.subkey=value" strings from generic --param CLI option.
        resolve_paths_base_dir: The directory relative to which paths in the 'paths' section
                                should be resolved. Defaults to CWD if None.

    Returns:
        The final, merged, path-resolved, and templated configuration dictionary.

    Raises:
        FileNotFoundError: If the default or specified config file cannot be found.
        yaml.YAMLError: If a config file is invalid YAML.
    """
    # 1. Load Default Config
    default_cfg_path = _get_default_config_path()
    if not os.path.exists(default_cfg_path):
        # Try finding relative to current working directory as fallback
        default_cfg_path_cwd = os.path.join(os.getcwd(), "drwiggle", DEFAULT_CONFIG_FILENAME)
        if os.path.exists(default_cfg_path_cwd):
             default_cfg_path = default_cfg_path_cwd
             logger.debug(f"Found default config relative to CWD: {default_cfg_path}")
        else:
             raise FileNotFoundError(f"Default config file not found at package location ({default_cfg_path}) or relative to CWD.")

    try:
        with open(default_cfg_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None: config = {} # Handle empty default file
        logger.debug(f"Loaded default config from {default_cfg_path}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing default config file {default_cfg_path}: {e}")
        raise

    # 2. Overlay User Config
    if config_path:
        abs_config_path = os.path.abspath(config_path)
        if not os.path.exists(abs_config_path):
            raise FileNotFoundError(f"User config file not found: {abs_config_path}")
        try:
            with open(abs_config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            if user_config: # Check if file is not empty and YAML is valid dict
                 config = deep_merge(config, user_config)
                 logger.debug(f"Merged user config from {abs_config_path}")
            else:
                 logger.warning(f"User config file {abs_config_path} is empty or invalid YAML dictionary.")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing user config file {abs_config_path}: {e}")
            raise

    # 3. Overlay Environment Variables
    env_config = _get_env_var_config()
    if env_config:
        config = deep_merge(config, env_config)
        logger.debug("Merged configuration from environment variables.")

    # 4. Overlay Specific CLI Overrides (processed by Click)
    if cli_overrides:
         config = deep_merge(config, cli_overrides)
         logger.debug(f"Merged configuration from specific CLI options: {cli_overrides}")

    # 5. Overlay Generic CLI --param Overrides
    cli_param_config = _parse_param_overrides(param_overrides)
    if cli_param_config:
        config = deep_merge(config, cli_param_config)
        logger.debug("Merged configuration from CLI --param overrides.")

    # --- Post-Merge Processing ---

    # 6. Handle Temperature Templating BEFORE resolving paths
    # Determine the temperature to use *after* all overrides
    current_temp = config.get("temperature", {}).get("current") # Might be None if not set
    if current_temp is None:
         # Fallback or default logic if temperature is crucial and not set
         default_temp = 320 # Example default
         config.setdefault("temperature", {})["current"] = default_temp
         current_temp = default_temp
         logger.warning(f"Temperature not specified, defaulting to {current_temp} for templating.")
    else:
         logger.info(f"Using temperature '{current_temp}' for configuration templating.")

    config = _template_config(config, current_temp)

    # 7. Resolve Paths (make paths absolute) AFTER templating
    config = _resolve_paths(config, base_dir=resolve_paths_base_dir)

    # 8. Validate Final Config (Basic Checks)
    required_sections = ['paths', 'binning', 'dataset', 'models', 'evaluation', 'temperature', 'pdb', 'system']
    for section in required_sections:
        if section not in config:
            logger.warning(f"Configuration section '{section}' is missing. Defaults may not apply.")
        elif not isinstance(config[section], dict):
             logger.warning(f"Configuration section '{section}' is not a dictionary. Check YAML structure.")

    # 9. Set Logging Level based on final config
    log_level_str = config.get("system", {}).get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, None)
    if log_level is None:
         log_level = logging.INFO
         logger.warning(f"Invalid log level '{log_level_str}' in config. Defaulting to INFO.")

    # Set level for the root logger AND the package logger
    logging.getLogger().setLevel(log_level)
    logging.getLogger('drwiggle').setLevel(log_level)
    logger.info(f"Logging level set to {log_level_str}")

    # Suppress excessive logging from dependencies if DEBUG is not set
    if log_level > logging.DEBUG:
        warnings.filterwarnings("ignore", category=FutureWarning) # Suppress some common warnings
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("numexpr").setLevel(logging.WARNING) # Often noisy
        # Add others as needed

    return config

# --- Helper functions to access specific config values ---

def get_metric_list(config: Dict[str, Any]) -> List[str]:
    """Extracts the list of enabled evaluation metrics from the config."""
    eval_config = config.get("evaluation", {}).get("metrics", {})
    return [metric for metric, enabled in eval_config.items() if enabled]

def get_visualization_colors(config: Dict[str, Any]) -> Dict[int, str]:
    """Extracts the color map for visualization from the config."""
    vis_config = config.get("visualization", {}).get("colors", {})
    try:
        # Convert keys to integers, ensure values are strings
        return {int(k): str(v) for k, v in vis_config.items()}
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not parse visualization colors correctly: {e}. Returning empty dict.")
        return {}

def get_class_names(config: Dict[str, Any]) -> Dict[int, str]:
    """Extracts the class names map from the config."""
    eval_config = config.get("evaluation", {}).get("class_names", {})
    try:
        # Convert keys to integers, ensure values are strings
        return {int(k): str(v) for k, v in eval_config.items()}
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not parse class names correctly: {e}. Returning empty dict.")
        return {}

def get_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Gets the specific config for a model, merging common settings."""
    models_config = config.get("models", {})
    common_config = models_config.get("common", {})
    model_specific_config = models_config.get(model_name, {})

    if not model_specific_config:
         logger.warning(f"No specific config found for model '{model_name}'. Using common settings only.")
         # Return a copy of common to avoid modifying it if specific is empty
         return copy.deepcopy(common_config)

    # Deep merge: model-specific overrides common
    merged_config = copy.deepcopy(common_config)
    merged_config = deep_merge(merged_config, model_specific_config)
    return merged_config

def get_feature_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Gets the feature configuration section."""
    return config.get("dataset", {}).get("features", {})

def get_enabled_features(config: Dict[str, Any]) -> List[str]:
    """Gets the list of features enabled for model input based on 'use_features'."""
    feature_cfg = get_feature_config(config)
    use_features_cfg = feature_cfg.get("use_features", {})
    return [feature for feature, enabled in use_features_cfg.items() if enabled]

def get_window_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Gets the window feature configuration section."""
    feature_cfg = get_feature_config(config)
    return feature_cfg.get("window", {})

def is_pdb_enabled(config: Dict[str, Any]) -> bool:
    """Checks if PDB processing is enabled."""
    return config.get("pdb", {}).get("enabled", False)

def get_pdb_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Gets the PDB configuration section."""
    return config.get("pdb", {})

def get_pdb_feature_config(config: Dict[str, Any]) -> Dict[str, bool]:
    """Gets the configuration for which features to extract from PDB."""
    if not is_pdb_enabled(config):
        return {}
    return config.get("pdb", {}).get("features", {})

def get_system_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Gets the system configuration section."""
    return config.get("system", {})

def get_binning_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Gets the binning configuration section."""
    return config.get("binning", {})

def get_split_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Gets the data splitting configuration section."""
    return config.get("dataset", {}).get("split", {})

def get_temperature_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Gets the temperature configuration section."""
    return config.get("temperature", {})