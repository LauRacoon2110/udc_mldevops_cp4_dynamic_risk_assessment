"""
Module for performing various diagnostics on the deployed model and data.

This module includes functions to:
- Make predictions using the deployed model.
- Calculate summary statistics for the dataset.
- Identify missing values in the dataset.
- Measure execution time for key processes.
- Check for outdated Python packages.
"""

import os
import pickle
import subprocess
import timeit
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import pkg_resources
from numpy.typing import NDArray

from utils import get_logger, load_config

# Initialize logger
logger = get_logger()

# Load configuration
root_path = Path(os.getcwd())
config = load_config(str(Path(root_path, "config.json")))

# Set environment variables
ENV = config["active_environment"]
deployment_path = Path(root_path, config[ENV]["deployment_path"])
test_data_path = Path(root_path, config[ENV]["test_data_path"])
output_folder_path = Path(root_path, config[ENV]["output_folder_path"])


def model_predictions(test_data: pd.DataFrame) -> NDArray[np.int_]:
    """
    Generate predictions using the deployed logistic regression model.

    Args:
        test_data (pd.DataFrame): The input dataset containing feature columns.

    Returns:
        NDArray[np.int_]: A numpy array containing binary predictions (0 = false, 1 = true).

    Raises:
        FileNotFoundError: If the deployed model file is not found.
        ValueError: If the number of predictions does not match the number of rows in the input dataset.
    """
    model_file_path = Path(deployment_path, "trainedmodel.pkl")
    if not model_file_path.exists():
        logger.error("Deployed model file not found at %s", model_file_path)
        raise FileNotFoundError(f"Deployed model file not found at {model_file_path}")

    with open(model_file_path, "rb") as model_file:
        trained_model = pickle.load(model_file)

    features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    X = test_data[features].values

    logger.info("Start prediction with deployed Logistic Regression model")
    y_pred: NDArray[np.int_] = trained_model.predict(X)

    if len(test_data) != len(y_pred):
        logger.error("Length of predictions does not match the number of rows in the input dataset")
        raise ValueError("Length of predictions does not match the number of rows in the input dataset")

    logger.info("Prediction completed with %s predictions", len(y_pred))
    logger.info(y_pred)
    return y_pred


def dataframe_summary() -> List[float]:
    """
    Calculate summary statistics (mean, median, std) for numeric columns in the dataset.

    Returns:
        List[float]: A flattened list containing summary statistics for each numeric column.

    Raises:
        FileNotFoundError: If the dataset file is not found.
    """
    data_file_path = Path(output_folder_path, "finaldata.csv")
    if not data_file_path.exists():
        logger.error("Final dataset file not found at %s", data_file_path)
        raise FileNotFoundError(f"Final dataset file not found at {data_file_path}")

    df = pd.read_csv(data_file_path, low_memory=False)
    numeric_cols = df.select_dtypes(include=np.number).columns

    summary_stats = df[numeric_cols].agg(["mean", "median", "std"])
    summary_stats_list: List[float] = summary_stats.values.flatten().tolist()
    logger.info("Summary statistics calculated for numeric columns: %s", numeric_cols)

    return summary_stats_list


def missing_values_summary() -> List[float]:
    """
    Calculate the percentage of missing values for each column in the dataset.

    Returns:
        List[float]: A list containing the percentage of missing values for each column.

    Raises:
        FileNotFoundError: If the dataset file is not found.
    """
    data_file_path = Path(output_folder_path, "finaldata.csv")
    if not data_file_path.exists():
        logger.error("Final dataset file not found at %s", data_file_path)
        raise FileNotFoundError(f"Final dataset file not found at {data_file_path}")

    df = pd.read_csv(data_file_path, low_memory=False)
    missing_values = df.isna().sum()
    missing_percentages = (missing_values / len(df)) * 100
    missing_percentages_list: List[float] = missing_percentages.tolist()
    logger.info("Missing values percentage calculated for each column")

    return missing_percentages_list


def execution_time() -> List[float]:
    """
    Measure execution time for data ingestion and model training scripts.

    Returns:
        List[float]: A list containing execution times (in seconds) for ingestion and training scripts.
    """
    ingest_cmd = f'python3 {Path(root_path, "ingestion.py")}'
    train_cmd = f'python3 {Path(root_path, "training.py")}'

    ingest_start = timeit.default_timer()
    os.system(ingest_cmd)
    ingest_duration = timeit.default_timer() - ingest_start

    train_start = timeit.default_timer()
    os.system(train_cmd)
    train_duration = timeit.default_timer() - train_start

    logger.info("Execution time for ingestion: %s seconds", ingest_duration)
    logger.info("Execution time for training: %s seconds", train_duration)

    return [ingest_duration, train_duration]


def outdated_packages_list() -> Dict[str, Union[str, List[Dict[str, str]]]]:
    """
    Check for outdated Python packages.

    Returns:
        dict: Contains both formatted text for logging and structured data for APIs.
    """
    logger.info("Generating list of outdated packages...\nThis might take a while ...")

    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    data = []

    for pkg, version in installed_packages.items():
        try:
            output = subprocess.check_output(["pip", "index", "versions", pkg], stderr=subprocess.DEVNULL, text=True)
            latest_version = next(
                (line.split()[-1] for line in output.splitlines() if line.startswith("  LATEST:")), "Unknown"
            )
        except subprocess.CalledProcessError:
            latest_version = "Not found"

        data.append({"package_name": pkg, "installed_version": version, "latest_version": latest_version})

    # Format as table for logging / console
    max_pkg_len = max(len(pkg["package_name"]) for pkg in data)
    max_installed_len = len("Installed Version")
    max_latest_len = len("Latest Version")
    header = (
        f"{'Package Name'.ljust(max_pkg_len)} | "
        f"{'Installed Version'.ljust(max_installed_len)} | "
        f"{'Latest Version'.ljust(max_latest_len)}"
    )
    output_lines = ["-" * len(header), header, "-" * len(header)]
    output_lines.extend(
        f"{pkg['package_name'].ljust(max_pkg_len)} | "
        f"{pkg['installed_version'].ljust(max_installed_len)} | "
        f"{pkg['latest_version'].ljust(max_latest_len)}"
        for pkg in sorted(data, key=lambda x: x["package_name"])
    )
    output_lines.append("-" * len(header))
    formatted = "\n".join(output_lines)
    logger.info("Package list generated successfully:")
    logger.info(formatted)

    return {"formatted": formatted, "structured": data}


if __name__ == "__main__":
    test_data_df = pd.read_csv(Path(test_data_path, "testdata.csv"), low_memory=False)

    model_predictions(test_data_df)
    dataframe_summary()
    missing_values_summary()
    execution_time()
    outdated_packages_list()
