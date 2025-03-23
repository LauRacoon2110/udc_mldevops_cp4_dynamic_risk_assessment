"""
This script trains a logistic regression model using data from the specified CSV file.
The trained model is then saved to a file for later use.
"""

import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils import get_logger, load_config

# Initialize logger
logger = get_logger()

# Load configuration
root_path = Path(os.getcwd())
config = load_config(str(Path(root_path, "config.json")))

# Set environment variables
ENV = config["active_environment"]
source_file = "finaldata.csv"
dataset_csv_path = Path(root_path, config[ENV]["output_folder_path"], source_file)
model_path = Path(root_path, config[ENV]["output_model_path"])


def train_model() -> None:
    """
    Train a logistic regression model using the data from the specified CSV file.
    The trained model is then saved to a file for later use.

    Returns:
        None
    """
    # Load the training data
    if not dataset_csv_path.exists():
        logger.error("Dataset file not found at %s", dataset_csv_path)
        raise FileNotFoundError(f"Dataset file not found at {dataset_csv_path}")

    train_data = pd.read_csv(dataset_csv_path, low_memory=False)
    logger.info("Loaded training data from %s", dataset_csv_path)

    # Hardcoded features and target
    features = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    target = "exited"

    # Extract features and target
    try:
        X = train_data[features].values
        y = train_data[target].values.ravel()
    except KeyError as e:
        logger.error("Missing required columns in the dataset: %s", str(e))
        raise KeyError(f"Missing required columns in the dataset: {str(e)}")

    logger.info("Starting training of Logistic Regression model")
    logger.info("Features used for training: %s", features)
    logger.info("Target used for training: %s", target)

    # Initialize and train the logistic regression model
    lr = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="auto",
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    model = lr.fit(X, y)
    logger.info("Training of Logistic Regression model complete")

    # Save the trained model
    model_file_path = Path(model_path, "trainedmodel.pkl")
    model_path.mkdir(parents=True, exist_ok=True)  # Ensure the model directory exists
    with open(model_file_path, "wb") as file_handler:
        pickle.dump(model, file_handler)
    logger.info("Model saved to %s", model_file_path)


if __name__ == "__main__":
    train_model()
