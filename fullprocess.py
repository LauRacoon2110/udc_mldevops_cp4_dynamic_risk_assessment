"""
Full process pipeline for dynamic risk assessment.

This script performs the following steps:
1. Checks for new data files in the source folder.
2. Ingests new data files into the system.
3. Checks for model drift by comparing the deployed model's score with the new score.
4. Retrains the model if model drift is detected.
5. Redeploys the model and associated artifacts.
6. Starts the Flask application and runs API calls.
7. Generates a confusion matrix for the model's performance.

Modules used:
- ingestion.py: Handles data ingestion.
- scoring.py: Scores the model.
- training.py: Trains the model.
- deployment.py: Deploys the model and artifacts.
- reporting.py: Generates reports.
- diagnostics.py: Performs diagnostics.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Set

import deployment
import ingestion
import reporting
import scoring
import training
from utils import get_logger, load_config

# Initialize logger
logger = get_logger()

# Load configuration
root_path = Path(os.getcwd())
config = load_config(str(Path(root_path, "config.json")))

# Set environment variables
ENV = config["active_environment"]

input_folder_path = Path(root_path, config[ENV]["input_folder_path"])
output_folder_path = Path(root_path, config[ENV]["output_folder_path"])
deployment_path = Path(root_path, config[ENV]["deployment_path"])
latest_score_path = Path(deployment_path, "latestscore.txt")
ingested_files_path = Path(output_folder_path, "ingestedfiles.txt")
test_data_file_path = Path(output_folder_path, "finaldata.csv")


def check_new_data() -> Optional[Set[str]]:
    """
    Check for new data files in the source folder.

    Returns:
        Optional[Set[str]]: A set of new data file names that are not yet ingested.
    """
    # Check if ingestedfiles.txt exists
    if not ingested_files_path.exists():
        logger.info(f"{ingested_files_path} does not exist. No data has been ingested so far.")
        # Return all files in the input folder as new files
        new_files = {file.name for file in input_folder_path.iterdir() if file.suffix == ".csv"}
        return new_files

    # If ingestedfiles.txt exists, read its contents
    with open(ingested_files_path, "r", encoding="utf-8") as f:
        ingested_files = {line.split(", ")[1].strip() for line in f.readlines()}

    # Compare files in the input folder with ingested files
    source_files = {file.name for file in input_folder_path.iterdir() if file.suffix == ".csv"}
    new_files = source_files - ingested_files

    return new_files


def check_model_drift() -> bool:
    """
    Check for model drift by comparing the latest deployed model's score with the new score.

    Returns:
        bool: True if model drift is detected, False otherwise.
    """
    # Step 1: Read the latest score from latestscore.txt
    deployed_score: Optional[float] = None
    if latest_score_path.exists():
        with open(latest_score_path, "r", encoding="utf-8") as f:
            last_line = f.read().strip()
            try:
                # Assuming the format is "timestamp, score"
                _, deployed_score_str = last_line.split(", ")
                deployed_score = float(deployed_score_str)
            except ValueError:
                logger.error(f"Invalid format in {latest_score_path}: {last_line}")
                raise ValueError(f"Invalid format in {latest_score_path}: {last_line}")
        logger.info(f"Deployed model score: {deployed_score}")
    else:
        logger.info("No latest score found from the already deployed model.")

    # Step 2: Use the deployed model to make predictions on the new data
    new_score = scoring.score_model(model_path=deployment_path, test_data=test_data_file_path)
    logger.info(f"New model score: {new_score}")

    # Step 3: Compare the scores
    if deployed_score is not None and new_score < deployed_score:
        logger.info(
            f"Model drift detected: New score ({new_score}) is lower than the deployed score ({deployed_score})."
        )
        return True
    else:
        logger.info(
            f"No model drift detected: New score ({new_score}) is equal to or higher "
            f"than the deployed score ({deployed_score})."
        )
        return False


def main() -> None:
    """
    Execute the full process pipeline for dynamic risk assessment.

    Steps:
    1. Check for new data files.
    2. Ingest new data files.
    3. Check for model drift.
    4. Retrain the model if drift is detected.
    5. Redeploy the model and artifacts.
    6. Start the Flask application and run API calls.
    7. Generate a confusion matrix for the model's performance.

    Returns:
        None
    """
    # Step 1: Check and read new data
    new_files = check_new_data()
    if not new_files:
        print("No new data found. Exiting process.")
        return

    # Step 2: Ingest new data
    print(f"New data files found: {new_files}. Proceeding with ingestion.")
    ingestion.merge_multiple_data_sources(files_to_ingest=new_files)

    # Step 3: Check for model drift
    if not check_model_drift():
        print("No model drift detected. Exiting process.")
        return  # Terminate if no model drift is detected

    # Step 4: Re-training
    print("Model drift detected. Proceeding with re-training.")
    training.train_model()

    # Step 5: Re-deployment
    print("Re-training completed. Proceeding with re-deployment.")
    deployment.store_inference_pipe_artifacts()

    # Step 6: Start Flask app and run apicalls.py
    print("Starting Flask application...")
    flask_process = subprocess.Popen(["python3", "app.py"])
    time.sleep(5)  # Wait for the Flask app to start

    try:
        print("Running API calls...")
        subprocess.run(["python3", "apicalls.py"], check=True)
    finally:
        print("Stopping Flask application...")
        flask_process.terminate()

    # Step 7: Reporting
    print("Generating confusion matrix...")
    reporting.score_model()


if __name__ == "__main__":
    main()
