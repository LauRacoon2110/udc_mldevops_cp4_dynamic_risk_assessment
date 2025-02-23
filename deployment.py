"""
This script handles the deployment of inference pipeline artifacts.
It copies the necessary files to the deployment directory.
"""

# pylint: disable=C0103  # Reason: Allowing non-conventional variable
# names for simplicity
import os
from pathlib import Path
import shutil
import json
import logging

# future imports from template
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# import pickle
# import pandas as pd
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
ingest_output_path = Path(os.getcwd(), config[ENV]["output_folder_path"])
model_output_path = Path(os.getcwd(), config[ENV]["output_model_path"])
deployment_path = Path(os.getcwd(), config[ENV]["deployment_path"])


def store_inference_pipe_artifacts():
    """
    Copy inference pipeline artifacts to the deployment directory.
    The artifacts include the latest score, trained model, and ingested files.
    """
    artifact_file_paths = [Path(model_output_path, 'latestscore.txt'),
                           Path(model_output_path, 'trainedmodel.pkl'),
                           Path(ingest_output_path, 'ingestedfiles.txt')]

    # Ensure the deployment directory exists
    deployment_path.mkdir(parents=True, exist_ok=True)

    for artifact in artifact_file_paths:
        if artifact.exists():
            shutil.copy(artifact, deployment_path)
            logger.info(
                "Copied inference artifact from %s to %s",
                artifact,
                deployment_path)
        else:
            logger.warning(
                "Source inference artifact on path %s does not exist",
                artifact)


if __name__ == '__main__':
    store_inference_pipe_artifacts()
