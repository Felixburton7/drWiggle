#!/usr/bin/env python3
"""
Configuration handling for the mdCATH processor pipeline.
Loads and validates YAML configuration file.
"""

import os
import yaml
import logging
from typing import Dict, List, Optional, Any, Union

# Default configuration values
DEFAULT_CONFIG = {
    "input": {
        "mdcath_dir": "/mnt/datasets/MD_CATH/data/",
        "domain_ids": [],  # Empty means process all domains
        "temperatures": [320, 348, 379, 413, 450],
    },
    "output": {
        "base_dir": os.path.expanduser("~/drFelix/data/"),
        "pdb_frames_dir": "interim/aposteriori_extracted_pdb_frames_files",
        "rmsf_dir": "interim/per-residue-rmsf",
        "summary_dir": "processed/mdcath_summary",
        "log_file": "pipeline.log",
    },
    "processing": {
        "frame_selection": {
            "method": "regular",  # Options: regular, rmsd, gyration, random
            "num_frames": 10,     # Number of frames to extract per temperature
            "cluster_method": "kmeans",  # For RMSD-based selection
        },
        "cleaning": {
            "replace_chain_0_with_A": True,
            "fix_atom_numbering": True,
            "remove_hydrogens": False,
        },
        "validation": {
            "check_missing_residues": True,
            "verify_rmsf_mapping": True,
            "compare_to_reference": False,
            "reference_path": "",
        }
    },
    "performance": {
        "num_cores": 0,  # 0 means auto-detect
        "batch_size": 10,
        "memory_limit_gb": 16,
    },
    "logging": {
        "level": "INFO",     # DEBUG, INFO, WARNING, ERROR, CRITICAL
        "console_level": "INFO", 
        "file_level": "DEBUG",
        "show_progress_bars": True,
    }
}

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file with fallback to default values.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dict containing merged configuration
    """
    logger = logging.getLogger(__name__)
    
    # Start with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Load user configuration if exists
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    # Deep merge configs (simple implementation)
                    deep_merge(config, user_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            logger.info(f"Using default configuration")
    else:
        logger.warning(f"Configuration file {config_path} not found. Using default configuration.")
    
    # Process and validate configuration
    config = process_config(config)
    
    return config

def process_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process and validate configuration:
    - Expand user paths
    - Set auto-detected values
    - Validate settings
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Processed configuration dictionary
    """
    # Expand user paths
    for key in ["mdcath_dir", "base_dir"]:
        if key == "mdcath_dir" and "input" in config:
            config["input"]["mdcath_dir"] = os.path.expanduser(config["input"]["mdcath_dir"])
        elif key == "base_dir" and "output" in config:
            config["output"]["base_dir"] = os.path.expanduser(config["output"]["base_dir"])
    
    # Auto-detect number of cores if not specified
    if config["performance"]["num_cores"] == 0:
        import multiprocessing
        config["performance"]["num_cores"] = multiprocessing.cpu_count()
    
    # Ensure all required directories are present
    for dir_key in ["pdb_frames_dir", "rmsf_dir", "summary_dir"]:
        if dir_key in config["output"]:
            # Make the path relative to base_dir if not absolute
            if not os.path.isabs(config["output"][dir_key]):
                config["output"][dir_key] = os.path.join(
                    config["output"]["base_dir"], 
                    config["output"][dir_key]
                )
    
    # Process log file path
    if "log_file" in config["output"] and not os.path.isabs(config["output"]["log_file"]):
        config["output"]["log_file"] = os.path.join(
            config["output"]["base_dir"],
            config["output"]["log_file"]
        )
    
    return config

def deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge source dict into target dict.
    
    Args:
        target: Target dictionary to merge into
        source: Source dictionary to merge from
        
    Returns:
        Merged dictionary (target is modified in-place)
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            deep_merge(target[key], value)
        else:
            target[key] = value
    return target

def create_example_config() -> None:
    """
    Create an example configuration file.
    """
    with open('config.yaml.example', 'w') as f:
        yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    # Generate example configuration file when run directly
    create_example_config()
    print("Created example configuration file: config.yaml.example")