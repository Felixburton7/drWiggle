#!/usr/bin/env python3
"""
Logging utilities for the mdCATH processor pipeline.
Sets up logging with file and console handlers, and progress tracking.
"""

import os
import sys
import logging
import datetime
from typing import Dict, Any, Optional
from tqdm import tqdm
import threading
import multiprocessing as mp
from multiprocessing import Value, Manager

# Configure global tracking variables for progress reporting
_progress_lock = None
_progress_dict = None
_total_domains = None
_processed_domains = None
_error_counts = None

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Configure logging based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Logger instance
    """
    log_file = config["output"]["log_file"]
    log_dir = os.path.dirname(log_file)
    
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Determine log levels
    console_level = getattr(logging, config["logging"]["console_level"])
    file_level = getattr(logging, config["logging"]["file_level"])
    root_level = min(console_level, file_level)  # Set to most verbose level
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(root_level)
    
    # Clear existing handlers (in case of reconfiguration)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s - %(funcName)s:%(lineno)d'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Log initialization
    logger.info(f"Logging initialized at {datetime.datetime.now()}")
    logger.info(f"Log file: {log_file}")
    
    return logger

def initialize_progress_tracking(total: int) -> None:
    """
    Initialize progress tracking for multiprocessing.
    
    Args:
        total: Total number of domains to process
    """
    global _progress_lock, _progress_dict, _total_domains, _processed_domains, _error_counts
    
    manager = Manager()
    _progress_lock = manager.Lock()
    _progress_dict = manager.dict()
    _total_domains = Value('i', total)
    _processed_domains = Value('i', 0)
    _error_counts = manager.dict({
        "missing_h5": 0,
        "extraction_error": 0,
        "cleaning_error": 0,
        "rmsf_error": 0,
        "other_error": 0
    })

def log_progress(domain_id: str, status: str, error_type: Optional[str] = None) -> None:
    """
    Update progress tracking for a domain.
    
    Args:
        domain_id: Domain identifier
        status: Status of processing (success, error, skipped)
        error_type: Type of error, if applicable
    """
    global _progress_dict, _processed_domains, _error_counts, _progress_lock
    
    if _progress_dict is not None:
        with _progress_lock:
            _progress_dict[domain_id] = status
            
            if status == "error" and error_type and error_type in _error_counts:
                _error_counts[error_type] += 1
            
            with _processed_domains.get_lock():
                _processed_domains.value += 1

def update_progress_bar() -> None:
    """
    Continuously update progress bar based on multiprocessing progress.
    Intended to be run in a separate thread.
    """
    global _progress_dict, _total_domains, _processed_domains, _progress_lock
    
    try:
        terminal_width = os.get_terminal_size().columns
    except OSError:
        terminal_width = 80
    
    bar_width = min(50, terminal_width - 30)
    start_time = datetime.datetime.now()
    
    while _processed_domains.value < _total_domains.value:
        current = _processed_domains.value
        progress = current / _total_domains.value
        filled_length = int(bar_width * progress)
        bar = '█' * filled_length + '-' * (bar_width - filled_length)
        percentage = int(100 * progress)
        
        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        domains_per_sec = current / max(elapsed, 0.1)
        remaining = _total_domains.value - current
        eta_seconds = remaining / max(domains_per_sec, 0.01)
        
        # Format ETA as HH:MM:SS
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        # Count by status
        with _progress_lock:
            success_count = sum(1 for status in _progress_dict.values() if status == "success")
            error_count = sum(1 for status in _progress_dict.values() if status == "error")
            skip_count = sum(1 for status in _progress_dict.values() if status == "skipped")
        
        sys.stdout.write('\r\033[K')  # Clear line
        sys.stdout.write(
            f"Progress: [{bar}] {percentage}% ({current}/{_total_domains.value}) "
            f"| Success: {success_count} | Errors: {error_count} | Skipped: {skip_count} "
            f"| ETA: {eta_str}"
        )
        sys.stdout.flush()
        
        # Sleep for a short time to avoid CPU hammering
        threading.Event().wait(0.5)
    
    # Final update
    sys.stdout.write('\r\033[K')  # Clear line
    sys.stdout.write(
        f"Completed: {_processed_domains.value}/{_total_domains.value} domains processed "
        f"in {datetime.datetime.now() - start_time}.\n"
    )
    sys.stdout.flush()

def start_progress_thread() -> threading.Thread:
    """
    Start the progress bar update thread.
    
    Returns:
        Thread object for the progress updater
    """
    progress_thread = threading.Thread(target=update_progress_bar)
    progress_thread.daemon = True
    progress_thread.start()
    return progress_thread

def get_progress_summary() -> Dict[str, Any]:
    """
    Get summary of processing progress.
    
    Returns:
        Dictionary with progress statistics
    """
    global _progress_dict, _total_domains, _processed_domains, _error_counts
    
    with _progress_lock:
        success_count = sum(1 for status in _progress_dict.values() if status == "success")
        error_count = sum(1 for status in _progress_dict.values() if status == "error")
        skip_count = sum(1 for status in _progress_dict.values() if status == "skipped")
        
        summary = {
            "total": _total_domains.value,
            "processed": _processed_domains.value,
            "success": success_count,
            "error": error_count,
            "skipped": skip_count,
            "error_counts": dict(_error_counts.items())
        }
    
    return summary

def log_info(message: str, logger: logging.Logger = None, domain_id: Optional[str] = None) -> None:
    """
    Log an info message, optionally with domain context.
    
    Args:
        message: Message to log
        logger: Logger instance (uses root logger if None)
        domain_id: Optional domain ID for context
    """
    if logger is None:
        logger = logging.getLogger()
    
    if domain_id:
        logger.info(f"[{domain_id}] {message}")
    else:
        logger.info(message)

def log_error(message: str, logger: logging.Logger = None, domain_id: Optional[str] = None, 
              error_type: Optional[str] = None, exc_info: bool = False) -> None:
    """
    Log an error message, optionally with domain context and error tracking.
    
    Args:
        message: Error message
        logger: Logger instance (uses root logger if None)
        domain_id: Optional domain ID for context
        error_type: Type of error for statistics
        exc_info: Whether to include exception info in log
    """
    if logger is None:
        logger = logging.getLogger()
    
    log_prefix = f"[{domain_id}] " if domain_id else ""
    logger.error(f"{log_prefix}{message}", exc_info=exc_info)
    
    if domain_id and error_type:
        log_progress(domain_id, "error", error_type)