"""
This script is used to report the performance of the deployed LR model
by reporting the confusion matrix.
"""

# pylint: disable=C0103  # Reason: Allowing non-conventional variable
# names for simplicity
import os
from pathlib import Path
import json
import logging

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from diagnostics import model_predictions

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
test_data_path = Path(os.getcwd(), config[ENV]["test_data_path"])
output_model_path = Path(os.getcwd(), config[ENV]["output_model_path"])


def score_model():
    """
    Calculate a confusion matrix using the test data and the deployed model.
    Plot and write this matrix to the workspace.
    """
    logger.info("Loading test data from %s", test_data_path)
    test_data_df = pd.read_csv(
        Path(
            test_data_path,
            "testdata.csv"),
        low_memory=False)

    logger.info("Extracting target variable 'exited' from test data")
    y_test = test_data_df.pop("exited")
    y_pred = model_predictions(test_data_df)

    logger.info("Calculating confusion matrix")
    confusion_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    logger.info("Raw confusion matrix values:\n%s", confusion_matrix)

    logger.info("Generating and storing the confusion matrix plot")
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=[
            'Not Excited', 'Excited'])

    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig(Path(output_model_path, 'confusionmatrix.png'))

    logger.info(
        "Confusion matrix plot saved to %s",
        Path(
            output_model_path,
            'confusionmatrix.png'))


# This block ensures that the reporting function is called only when this
# script is executed directly.
if __name__ == '__main__':
    score_model()
