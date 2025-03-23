"""
This script reports the performance of the deployed logistic regression model
by calculating and plotting the confusion matrix.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from diagnostics import model_predictions
from utils import get_logger, load_config

# Initialize logger
logger = get_logger()

# Load configuration
root_path = Path(os.getcwd())
config = load_config(str(Path(root_path, "config.json")))

# Set environment variables
ENV = config["active_environment"]
test_data_path = Path(root_path, config[ENV]["test_data_path"])
output_model_path = Path(root_path, config[ENV]["output_model_path"])


def score_model() -> None:
    """
    Calculate a confusion matrix using the test data and the deployed model.
    Plot and save this matrix to the workspace.

    The function performs the following steps:
    1. Loads the test data from the configured path.
    2. Extracts the target variable (`exited`) from the test data.
    3. Generates predictions using the deployed model.
    4. Calculates the confusion matrix.
    5. Plots and saves the confusion matrix as an image.

    Returns:
        NoReturn: This function does not return any value.
    """
    # Load test data
    logger.info("Loading test data from %s", test_data_path)
    test_data_file = Path(test_data_path, "testdata.csv")
    if not test_data_file.exists():
        logger.error("Test data file not found at %s", test_data_file)
        raise FileNotFoundError(f"Test data file not found at {test_data_file}")

    test_data_df = pd.read_csv(test_data_file, low_memory=False)

    # Extract target variable
    logger.info("Extracting target variable 'exited' from test data")
    if "exited" not in test_data_df.columns:
        logger.error("Target variable 'exited' not found in test data")
        raise KeyError("Target variable 'exited' not found in test data")

    y_test = test_data_df.pop("exited")

    # Generate predictions
    logger.info("Generating predictions using the deployed model")
    y_pred = model_predictions(test_data_df)

    # Calculate confusion matrix
    logger.info("Calculating confusion matrix")
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    logger.info("Raw confusion matrix values:\n%s", conf_matrix)

    # Generate and save confusion matrix plot
    logger.info("Generating and storing the confusion matrix plot")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Not Exited", "Exited"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.subplots_adjust(left=0.2, bottom=0.2)

    output_file = Path(output_model_path, "confusionmatrix.png")
    plt.savefig(output_file)
    logger.info("Confusion matrix plot saved to %s", output_file)


if __name__ == "__main__":
    score_model()
