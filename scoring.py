"""
This script loads a trained logistic regression model and scores it using test data.
The F1 score of the model is then saved to a file for later use.
"""

# pylint: disable=C0103  # Reason: Allowing non-conventional variable
# names for simplicity
import os
from pathlib import Path
import pickle
import json
from datetime import datetime
import logging
import pandas as pd

from sklearn import metrics

# future imports from template
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# import numpy as np
# from flask import Flask, session, jsonify, request


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
output_model_path = Path(os.getcwd(), config[ENV]["output_model_path"])
test_data_path = Path(os.getcwd(), config[ENV]["test_data_path"])


def score_model():
    """
    Load the trained logistic regression model and score it using the test data.
    The F1 score is calculated and saved to a file.
    """
    with open(Path(output_model_path, "trainedmodel.pkl"), "rb") as model_file:
        trained_model = pickle.load(model_file)
    test_data = pd.read_csv(
        Path(
            test_data_path,
            "testdata.csv"),
        low_memory=False)

    features = [
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees"]
    target = ["exited"]

    X = test_data[features].values.reshape(-1, 3)
    y = test_data[target].values.reshape(-1, 1).ravel()

    logger.info("Start prediction and scoring with Logistic Regression model")
    logger.info(
        "Model for scoring: %s",
        Path(output_model_path, "trainedmodel.pkl"))
    logger.info("Test data for scoring: %s",
                Path(test_data_path, "testdata.csv"))

    y_pred = trained_model.predict(X)
    f1 = metrics.f1_score(y, y_pred)
    logger.info("Prediction and scoring completed with F1 score: %s", f1)

    with open(Path(output_model_path, "latestscore.txt"), "w", encoding="utf-8") as log_file:
        scoring_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{scoring_time}, {f1}\n")
        logger.info(
            "Scoring details for the latest test run logged to: %s",
            Path(output_model_path, "latestscore.txt"))


# This block ensures that the score_model function is called only when this
# script is executed directly.
if __name__ == "__main__":
    score_model()
