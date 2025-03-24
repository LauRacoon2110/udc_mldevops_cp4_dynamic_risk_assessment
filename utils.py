"""
Utility functions for logging, configuration management, and file path handling.

This module provides:
1. A function to initialize and configure a logger.
2. A function to load configuration settings from a JSON file.
3. A function to generate a unique file path by appending a number if the file already exists.
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
        logging.Logger: A configured logger object for logging messages.
    """
    logger = logging.getLogger(__name__)  # Use the module's name as the logger name
    logger.setLevel(logging.INFO)

    # Define the log message format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Configure the stream handler to output logs to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # Add the handler to the logger if not already added
    if not logger.handlers:  # Avoid adding multiple handlers if the logger is reused
        logger.addHandler(stream_handler)

    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration settings from a JSON file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration settings.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        json.JSONDecodeError: If the configuration file is not a valid JSON file.
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config: Dict[str, Any] = json.load(file)
        return config


def get_unique_file_path(base_path: Path, file_name: str) -> Path:
    """
    Generate a unique file path by appending a number to the file name if it already exists.

    For example:
    - If "output.txt" exists, the function will return "output_1.txt".
    - If "output_1.txt" also exists, it will return "output_2.txt", and so on.

    Args:
        base_path (Path): The directory where the file will be saved.
        file_name (str): The desired file name.

    Returns:
        Path: A unique file path that does not already exist in the specified directory.
    """
    file_path = base_path / file_name
    counter = 2

    while file_path.exists():
        stem, suffix = file_name.rsplit(".", 1)
        file_path = base_path / f"{stem}{counter}.{suffix}"
        counter += 1

    return file_path
