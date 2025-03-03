"""
Module for performing various diagnostics on the deployed model and data.
"""

# pylint: disable=C0103  # Reason: Allowing non-conventional variable
# names for simplicity
import os
import subprocess
from pathlib import Path
from typing import List
import json
import pickle
import logging

import timeit
import pkg_resources
import pandas as pd
import numpy as np


# Configure logging to output to the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Load config.json and set VARs
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

ENV = config["active_environment"]
deployment_path = Path(os.getcwd(), config[ENV]["deployment_path"])
test_data_path = Path(os.getcwd(), config[ENV]["test_data_path"])
output_folder_path = Path(os.getcwd(), config[ENV]["output_folder_path"])


def model_predictions(test_data: pd.DataFrame) -> List[int]:
    """
    Return predictions made by the deployed model.

    This function takes an argument that consists of a dataset, in a pandas DataFrame format.
    It reads the deployed model from the directory specified in the prod_deployment_path entry
    of your config.json file.
    The function uses the deployed model to make predictions for each row of the input dataset.

    Args:
        test_data (pd.DataFrame): The input dataset for which predictions are to be made.

    Returns:
        List[int]: A list of binary predictions (0 = false, 1 = true).
                   This list has the same length as the number of rows in the input dataset.
    """

    # read deployed model
    with open(Path(deployment_path, "trainedmodel.pkl"), "rb") as model_file:
        trained_model = pickle.load(model_file)

    features = [
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees"]

    X = test_data[features].values.reshape(-1, 3)

    logger.info("Start prediction with deployed Logistic Regression model")
    logger.info(
        "Model for prediction: %s",
        Path(deployment_path, "trainedmodel.pkl"))
    logger.info("Test dataset for prediction: %s",
                Path(test_data_path, "testdata.csv"))

    y_pred = trained_model.predict(X)
    y_pred_list = y_pred.tolist()

    if len(test_data) != len(y_pred_list):
        logger.error(
            "Length of predictions does not match the number of rows in the input dataset")
        raise ValueError(
            "Length of predictions does not match the number of rows in the input dataset")
    logger.info("Number of predictions to make: %s", len(test_data))
    logger.info("Predictions (%s): %s", len(y_pred_list), y_pred_list)

    logger.info("Prediction completed\n")
    return y_pred


def dataframe_summary() -> List[float]:
    """
    Calculate summary statistics for the dataset stored in the directory
    specified by output_folder_path in config.json.
    The summary statistics calculated are
       * means
       * medians
       * standard deviations
    for each numeric column in the data.

    Returns:
        List[float]: A list containing all of the summary statistics for
                     every numeric column of the input dataset.
    """
    # read the final data
    df = pd.read_csv(
        Path(
            output_folder_path,
            "finaldata.csv"),
        low_memory=False)

    # get all numeric cols
    numeric_cols = df.select_dtypes(include=np.number).columns
    logger.info(
        "Numeric columns of the final dataset: %s", [
            x for x in numeric_cols])

    # for all numeric cols - calculate mean, median, std
    summary_stats = df[numeric_cols].agg(['mean', 'median', 'std'])
    logger.info("Summary statistics of the final dataset:\n%s", summary_stats)

    # create a list stats values
    summary_stats_list = summary_stats.values.flatten().tolist()

    logger.info("Summary statistics completed\n")
    return summary_stats_list


def missing_values_summary() -> List[int]:
    """
    Count the number of NA values in each column of the dataset stored in the directory
    specified by output_folder_path in config.json.
    Calculate the percentage of NA values for each column.

    Returns:
        List[float]: A list containing the percentage of NA values for each column in the dataset.
    """

    # get the final data
    df = pd.read_csv(
        Path(
            output_folder_path,
            "finaldata.csv"),
        low_memory=False)

    # get missing values
    missing_values = df.isna().sum()
    logger.info("Missing values in the final dataset:\n%s", missing_values)

    # create a list of missing values in percentage for each column
    missing_values_percentage_list = [
        missing_values[i] / len(df.index) for i in range(len(missing_values))]
    logger.info(
        "Missing values percentage per column in the final dataset:\n%s",
        missing_values_percentage_list)

    logger.info("Missing values statistics completed\n")
    return missing_values_percentage_list


# Function to get timings
def execution_time() -> List[float]:
    """
    Calculate the timing of important tasks: data ingestion and model training.

    This function times how long it takes to perform data ingestion (ingestion.py)
    and model training (training.py).

    Returns:
        List[float]: A list containing timing measurements in seconds
                     for data ingestion and model training)
    """
    ingest_cmd = f'python3 {Path(os.getcwd(), "ingestion.py")}'
    ingest_start = timeit.default_timer()
    os.system(ingest_cmd)
    ingest_duration = timeit.default_timer() - ingest_start
    logging.info("Execution time of data ingestion: %s", ingest_duration)

    train_cmd = f'python3 {Path(os.getcwd(), "training.py")}'
    train_start = timeit.default_timer()
    os.system(train_cmd)
    train_duration = timeit.default_timer() - train_start
    logging.info("Execution time of model training: %s", train_duration)

    return [ingest_duration, train_duration]


def outdated_packages_list() -> None:
    """
    Check the current and latest versions of all the modules used in the project.

    This function checks the current and latest versions of all
    the modules this project workspace uses.

    It outputs a table with three columns: the name of the module,
    the currently installed version, and the latest available version.

    Returns:
        None
    """

    # Get installed packages
    installed_packages = {
        pkg.key: pkg.version for pkg in pkg_resources.working_set}  # pylint: disable=E1133

    # Prepare data storage
    data = []

    for pkg, version in installed_packages.items():
        latest_version = "Unknown"

        try:
            output = subprocess.check_output(
                ["pip", "index", "versions", pkg],
                stderr=subprocess.DEVNULL,
                text=True
            )
            for line in output.splitlines():
                if line.startswith("  LATEST:"):
                    latest_version = line.split()[-1]  # Extract latest version
                    break
        except subprocess.CalledProcessError:
            latest_version = "Not found"  # Handle error if package lookup fails

        # Store row data
        data.append((pkg, version, latest_version))

    # Determine column widths
    max_pkg_len = max(len(pkg) for pkg, _, _ in data)
    max_installed_len = len("Installed Version")
    max_latest_len = len("Latest Version")

    # Print table header
    header = (
        f"{'Package Name'.ljust(max_pkg_len)} | "
        f"{'Installed Version'.ljust(max_installed_len)} | "
        f"{'Latest Version'.ljust(max_latest_len)}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # Print each row
    data.sort(key=lambda x: x[0])
    for pkg, installed, latest in data:
        print(
            f"{pkg.ljust(max_pkg_len)} | "
            f"{installed.ljust(max_installed_len)} | "
            f"{latest.ljust(max_latest_len)}"
        )

    print("-" * len(header))


if __name__ == '__main__':

    test_data_df = pd.read_csv(
        Path(
            test_data_path,
            "testdata.csv"),
        low_memory=False)

    model_predictions(test_data_df)
    dataframe_summary()
    missing_values_summary()
    execution_time()
    outdated_packages_list()
