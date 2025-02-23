"""
This script trains a logistic regression model using data from the specified CSV file.
The trained model is then saved to a file for later use.
"""

# pylint: disable=C0103  # Reason: Allowing non-conventional variable
# names for simplicity
import os
from pathlib import Path
import pickle
import json
import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression

# future imports from template
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
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
source_file = "finaldata.csv"
dataset_csv_path = Path(
    os.getcwd(),
    config[ENV]["output_folder_path"],
    source_file)
model_path = Path(os.getcwd(), config[ENV]["output_model_path"])


def train_model():
    """
    Train a logistic regression model using the data from the specified CSV file.
    The trained model is then saved to a file for later use.
    """
    train_data = pd.read_csv(dataset_csv_path, low_memory=False)
    features = [
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees"]
    target = ["exited"]

    X = train_data[features].values.reshape(-1, 3)
    y = train_data[target].values.reshape(-1, 1).ravel()

    logger.info("Starting training of Logistic Regression model")
    logger.info("Features used for training: %s", features)
    logger.info("Target used for training: %s", target)

    lr = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='auto',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False)

    model = lr.fit(X, y)
    logger.info("Training of Logistic Regression model complete")

    with open(Path(model_path, 'trainedmodel.pkl'), 'wb') as file_handler:
        pickle.dump(model, file_handler)
    logger.info("Model saved to %s", Path(model_path, 'trainedmodel.pkl'))


# This block ensures that the merge function is called only when this
# script is executed directly.
if __name__ == "__main__":
    train_model()
