"""
This script interacts with the Flask API endpoints to perform the following tasks:
- Generate predictions using the deployed model.
- Retrieve the F1 score of the deployed model.
- Get summary statistics for the dataset.
- Perform diagnostics on the system.

The responses from the API endpoints are combined and written to a file for further analysis.
"""

import json
import os
from pathlib import Path
from typing import Dict

import requests

from utils import get_logger, get_unique_file_path, load_config

# Initialize logger
logger = get_logger()

# Load configuration
root_path = Path(os.getcwd())
config = load_config(str(Path(root_path, "config.json")))

# Set environment variables
ENV = config["active_environment"]
output_model_path = Path(root_path, config[ENV]["output_model_path"])
test_data_path = Path(root_path, config[ENV]["test_data_path"])

# Specify the base URL of your Flask API
BASE_URL = "http://127.0.0.1:8000"

# API endpoints
PREDICTION_ENDPOINT = f"{BASE_URL}/prediction"
SCORING_ENDPOINT = f"{BASE_URL}/scoring"
SUMMARY_STATS_ENDPOINT = f"{BASE_URL}/summarystats"
DIAGNOSTICS_ENDPOINT = f"{BASE_URL}/diagnostics"


def call_api_endpoints() -> Dict[str, object]:
    """
    Call the Flask API endpoints and collect their responses.

    Returns:
        Dict[str, object]: A dictionary containing the responses from the API endpoints.
    """
    logger.info("Calling API endpoints...")

    try:
        # Call the prediction endpoint
        logger.info("Calling prediction endpoint...")
        prediction_response = requests.post(PREDICTION_ENDPOINT, params={"file_path": "testdata.csv"})
        prediction_response.raise_for_status()
        prediction_result = prediction_response.json()
        logger.info("Prediction endpoint response received.")

        # Call the scoring endpoint
        logger.info("Calling scoring endpoint...")
        scoring_response = requests.get(SCORING_ENDPOINT)
        scoring_response.raise_for_status()
        scoring_result = scoring_response.json()
        logger.info("Scoring endpoint response received.")

        # Call the summary statistics endpoint
        logger.info("Calling summary statistics endpoint...")
        summary_stats_response = requests.get(SUMMARY_STATS_ENDPOINT)
        summary_stats_response.raise_for_status()
        summary_stats_result = summary_stats_response.json()
        logger.info("Summary statistics endpoint response received.")

        # Call the diagnostics endpoint
        logger.info("Calling diagnostics endpoint...")
        diagnostics_response = requests.get(DIAGNOSTICS_ENDPOINT)
        diagnostics_response.raise_for_status()
        diagnostics_result = diagnostics_response.text
        logger.info("Diagnostics endpoint response received.")

        # Combine all responses
        responses = {
            "prediction": prediction_result,
            "scoring": scoring_result,
            "summary_stats": summary_stats_result,
            "diagnostics": diagnostics_result,
        }

        logger.info("All API responses collected successfully.")
        return responses

    except requests.exceptions.RequestException as e:
        logger.error("Error while calling API endpoints: %s", str(e))
        raise


def write_responses_to_file(responses: Dict[str, object], output_file_path: Path) -> None:
    """
    Write the API responses to a file in JSON format.

    Args:
        responses (Dict[str, object]): The dictionary containing API responses.
        output_file_path (Path): The path to the output file where responses will be written.

    Returns:
        None
    """
    logger.info("Writing API responses to file: %s", output_file_path)
    try:
        with open(output_file_path, "w", encoding="utf-8") as file:
            json.dump(responses, file, indent=4)
        logger.info("API responses written to file successfully.")
    except Exception as e:
        logger.error("Error while writing API responses to file: %s", str(e))
        raise


if __name__ == "__main__":
    """
    Main script execution:
    - Calls the API endpoints.
    - Writes the combined responses to a file.
    """
    try:
        # Call API endpoints and collect responses
        api_responses = call_api_endpoints()

        # Define the base file name
        base_file_name = "apireturns.txt"

        # Get a unique file path
        output_file = get_unique_file_path(output_model_path, base_file_name)

        # Write responses to the output file
        write_responses_to_file(api_responses, output_file)

        print("API responses have been written to apireturns.txt")
    except Exception as e:
        logger.error("An error occurred during the API calls or file writing process: %s", str(e))
        print("An error occurred. Check the logs for more details.")
