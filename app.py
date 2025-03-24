"""
Flask application for serving endpoints related to the deployed model.

This application provides the following endpoints:
- /prediction: Generate predictions using the deployed model.
- /scoring: Get the F1 score of the deployed model.
- /summarystats: Get summary statistics for the dataset.
- /diagnostics: Perform diagnostics on the system, including execution time, missing values, and outdated packages.
"""

import os
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from flask import Flask, Response, jsonify, request

from diagnostics import (
    dataframe_summary,
    execution_time,
    missing_values_summary,
    model_predictions,
    outdated_packages_list,
)
from scoring import score_model
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

# Initialize Flask app
app = Flask(__name__)


def read_file_to_predict(file_name: str) -> pd.DataFrame:
    """
    Read the file to predict from the test_data_path directory.

    Args:
        file_name (str): The name of the file to read.

    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist in the test_data_path directory.
    """
    file_path = Path(test_data_path, file_name)
    if not file_path.exists():
        logger.error("File not found at %s", file_path)
        raise FileNotFoundError(f"File not found at {file_path}")
    logger.info("Reading file for prediction: %s", file_path)
    return pd.read_csv(file_path, low_memory=False)


@app.route("/prediction", methods=["POST", "OPTIONS"])
def predict() -> Union[Response, Tuple[Response, int]]:
    if request.method == "OPTIONS":
        return jsonify({"methods": ["POST", "OPTIONS"]}), 200

    input_file_path = request.args.get("file_path")
    if not input_file_path:
        logger.error("Missing 'file_path' argument in the request")
        return jsonify({"error": "Missing 'file_path' argument"}), 400

    try:
        data_df = read_file_to_predict(input_file_path)
        y_pred = model_predictions(data_df)
        return jsonify(y_pred.tolist())
    except Exception as e:
        logger.error("Error during prediction: %s", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/scoring", methods=["GET", "OPTIONS"])
def score() -> Union[Response, Tuple[Response, int]]:
    if request.method == "OPTIONS":
        return jsonify({"methods": ["GET", "OPTIONS"]}), 200

    try:
        f1_score = score_model()
        return jsonify(f1_score)
    except Exception as e:
        logger.error("Error during scoring: %s", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/summarystats", methods=["GET", "OPTIONS"])
def sum_stats() -> Union[Response, Tuple[Response, int]]:
    if request.method == "OPTIONS":
        return jsonify({"methods": ["GET", "OPTIONS"]}), 200

    try:
        summary_stats = dataframe_summary()
        return jsonify(summary_stats)
    except Exception as e:
        logger.error("Error during summary statistics calculation: %s", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnose() -> Union[Response, Tuple[Response, int]]:
    if request.method == "OPTIONS":
        return jsonify({"methods": ["GET", "OPTIONS"]}), 200

    try:
        duration = execution_time()
        missing_values = missing_values_summary()
        pkg_dependencies = outdated_packages_list()
        return jsonify(
            {
                "execution_time": duration,
                "missing_values": missing_values,
                "outdated_packages": pkg_dependencies["structured"],
            }
        )
    except Exception as e:
        logger.error("Error during diagnostics: %s", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
