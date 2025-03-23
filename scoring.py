"""
This script loads a trained logistic regression model and scores it using test data.
The F1 score of the model is then saved to a file for later use.
"""

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from sklearn import metrics

from utils import get_logger, load_config

# Initialize logger
logger = get_logger()

# Load configuration
root_path = Path(os.getcwd())
config = load_config(str(Path(root_path, "config.json")))

# Set environment variables
ENV = config["active_environment"]
output_model_path = Path(root_path, config[ENV]["output_model_path"])
test_data_path = Path(root_path, config[ENV]["test_data_path"])


def score_model() -> float:
    """
    Load the trained logistic regression model and score it using the test data.
    The F1 score is calculated and saved to a file.

    Returns:
        float: The F1 score of the model on the test data.
    """
    # Load the trained model
    model_file_path = Path(output_model_path, "trainedmodel.pkl")
    if not model_file_path.exists():
        logger.error("Trained model file not found at %s", model_file_path)
        raise FileNotFoundError(f"Trained model file not found at {model_file_path}")

    with open(model_file_path, "rb") as model_file:
        trained_model = pickle.load(model_file)
    logger.info("Loaded trained model from %s", model_file_path)

    # Load the test data
    test_data_file_path = Path(test_data_path, "testdata.csv")
    if not test_data_file_path.exists():
        logger.error("Test data file not found at %s", test_data_file_path)
        raise FileNotFoundError(f"Test data file not found at {test_data_file_path}")

    test_data = pd.read_csv(test_data_file_path, low_memory=False)
    logger.info("Loaded test data from %s", test_data_file_path)

    # Hardcoded features and target
    features: List[str] = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    target: str = "exited"

    # Extract features and target
    try:
        X = test_data[features].values
        y = test_data[target].values.ravel()
    except KeyError as e:
        logger.error("Missing required columns in the test dataset: %s", str(e))
        raise KeyError(f"Missing required columns in the test dataset: {str(e)}")

    # Perform predictions and calculate F1 score
    logger.info("Start prediction and scoring with Logistic Regression model")
    y_pred = trained_model.predict(X)
    f1: float = metrics.f1_score(y, y_pred)
    logger.info("Prediction and scoring completed with F1 score: %s", f1)

    # Save the F1 score to a file
    latest_score_file_path = Path(output_model_path, "latestscore.txt")
    with open(latest_score_file_path, "w", encoding="utf-8") as log_file:
        scoring_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{scoring_time}, {f1}\n")
        logger.info("Scoring details for the latest test run logged to: %s", latest_score_file_path)

    return f1


if __name__ == "__main__":
    score_model()
