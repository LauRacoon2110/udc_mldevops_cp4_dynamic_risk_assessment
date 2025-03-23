"""
This script handles the deployment of inference pipeline artifacts.
It copies the necessary files (latest score, trained model, and ingested files)
to the deployment directory for production use.
"""

import os
import shutil
from pathlib import Path
from typing import List

from utils import get_logger, load_config

# Initialize logger
logger = get_logger()

# Load configuration
root_path = Path(os.getcwd())
config = load_config(str(Path(root_path, "config.json")))

# Set environment variables
ENV = config["active_environment"]
ingest_output_path = Path(root_path, config[ENV]["output_folder_path"])
model_output_path = Path(root_path, config[ENV]["output_model_path"])
deployment_path = Path(root_path, config[ENV]["deployment_path"])


def store_inference_pipe_artifacts() -> None:
    """
    Copy inference pipeline artifacts to the deployment directory.
    The artifacts include:
        - latestscore.txt: The latest F1 score of the model.
        - trainedmodel.pkl: The trained logistic regression model.
        - ingestedfiles.txt: The log of ingested files.

    The function ensures that the deployment directory exists and copies
    the artifacts from their respective locations to the deployment directory.

    Returns:
        None
    """
    artifact_file_paths: List[Path] = [
        Path(model_output_path, "latestscore.txt"),
        Path(model_output_path, "trainedmodel.pkl"),
        Path(ingest_output_path, "ingestedfiles.txt"),
    ]

    # Ensure the deployment directory exists
    deployment_path.mkdir(parents=True, exist_ok=True)
    logger.info("Deployment directory ensured at %s", deployment_path)

    for artifact in artifact_file_paths:
        if artifact.exists():
            shutil.copy(artifact, deployment_path)
            logger.info("Copied inference artifact from %s to %s", artifact, deployment_path)
        else:
            logger.warning("Source inference artifact does not exist at path: %s", artifact)


if __name__ == "__main__":
    store_inference_pipe_artifacts()
