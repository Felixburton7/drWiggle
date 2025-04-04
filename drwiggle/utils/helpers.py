import os
import logging
import time
from functools import wraps
from typing import Iterable, TypeVar, Optional, Dict, List, Any
import joblib # For saving/loading python objects easily
import glob
import traceback

try:
    from tqdm.auto import tqdm # Use richer progress bars if available (notebooks)
except ImportError:
    try:
        from tqdm import tqdm # Fallback to standard tqdm
    except ImportError:
        # Provide a dummy tqdm if not installed at all
        print("Warning: tqdm not installed. Progress bars will be disabled.")
        print("Install tqdm for progress bars: pip install tqdm")
        def tqdm(iterable, *args, **kwargs):
            return iterable

logger = logging.getLogger(__name__)

T = TypeVar('T') # Generic type variable

def timer(func):
    """Decorator to time function execution and log the duration."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        # Log duration at DEBUG level to avoid cluttering INFO logs
        logger.debug(f"Function '{func.__name__}' executed in {duration:.4f} seconds")
        return result
    return wrapper

def ensure_dir(directory_path: str):
    """
    Creates a directory if it doesn't exist. Logs success or failure.

    Args:
        directory_path: The path of the directory to create.
    """
    if not directory_path:
        logger.warning("ensure_dir called with empty or None path. Skipping.")
        return
    abs_path = os.path.abspath(directory_path)
    if not os.path.exists(abs_path):
        try:
            os.makedirs(abs_path, exist_ok=True)
            logger.debug(f"Created directory: {abs_path}")
        except OSError as e:
            logger.error(f"Failed to create directory {abs_path}: {e}")
            raise # Re-raise the exception after logging
    else:
         logger.debug(f"Directory already exists: {abs_path}")

# Wrapper for tqdm progress bar with optional disabling based on log level
def progress_bar(
    iterable: Iterable[T],
    desc: Optional[str] = None,
    total: Optional[int] = None,
    disable: Optional[bool] = None, # Allow explicit disabling
    leave: bool = True,
    **kwargs
) -> Iterable[T]:
    """
    Provides a tqdm progress bar with flexible disabling options.

    Disables the bar if log level is WARNING or higher, unless explicitly
    enabled via `disable=False`.

    Args:
        iterable: The iterable to wrap with a progress bar.
        desc: Optional description for the progress bar.
        total: Optional total number of items (useful if len(iterable) is slow/unavailable).
        disable: Explicitly disable (True) or enable (False) the bar. If None, uses log level.
        leave: Whether to leave the finished progress bar visible (default True).
        **kwargs: Additional arguments passed directly to tqdm.

    Returns:
        The tqdm-wrapped iterable.
    """
    if disable is None: # Automatic disabling based on log level
        log_level = logging.getLogger().getEffectiveLevel()
        # Disable if logging level is WARNING or above (more severe)
        effective_disable = log_level >= logging.WARNING
    else: # Respect explicit setting
        effective_disable = disable

    # Try to get total length if not provided and not disabled
    if total is None and not effective_disable:
        try:
            total = len(iterable) # type: ignore
        except (TypeError, AttributeError):
            total = None # Cannot determine length

    # Set default tqdm arguments if not provided
    kwargs.setdefault('ncols', 100) # Default width
    kwargs.setdefault('bar_format', '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    return tqdm(
        iterable,
        desc=desc,
        total=total,
        disable=effective_disable,
        leave=leave,
        **kwargs
    )

def save_object(obj: Any, path: str, compress: int = 3):
    """
    Saves a Python object to a file using joblib.

    Args:
        obj: The Python object to save.
        path: The file path (directory will be created).
        compress: Compression level for joblib (0-9). Default is 3.
    """
    try:
        ensure_dir(os.path.dirname(path))
        joblib.dump(obj, path, compress=compress)
        logger.info(f"Object saved successfully to {path} (compression={compress})")
    except Exception as e:
        logger.error(f"Failed to save object to {path}: {e}", exc_info=True)
        raise IOError(f"Could not save object to file: {path}") from e

def load_object(path: str) -> Any:
    """
    Loads a Python object from a file saved using joblib.

    Args:
        path: The file path.

    Returns:
        The loaded Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If loading fails.
    """
    if not os.path.exists(path):
        logger.error(f"File not found for loading object: {path}")
        raise FileNotFoundError(f"Cannot load object, file not found: {path}")
    try:
        obj = joblib.load(path)
        logger.info(f"Object loaded successfully from {path}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load object from {path}: {e}", exc_info=True)
        raise IOError(f"Could not load object from file: {path}") from e

# Add other general utility functions here as needed.
