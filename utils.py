"""
Utility functions for logging and configuration management.

This module provides:
1. A function to initialize and configure a logger.
2. A function to load configuration settings from a JSON file.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict


def get_logger() -> logging.Logger:
    """
    Initialize and configure a logger object.

    The logger outputs messages to the console with a specific format that includes
    the timestamp, logger name, log level, and message.

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(__name__)  # Use the module's name as the logger name
    logger.setLevel(logging.INFO)

    # Define the log message format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Configure the stream handler to output logs to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.handlers:  # Avoid adding multiple handlers if the logger is reused
        logger.addHandler(stream_handler)

    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration settings from a JSON file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict: Configuration dictionary containing the settings.
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as file:
        config: Dict[str, Any] = json.load(file)
        return config
