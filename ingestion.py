# pylint: disable=C0103  # Reason: Allowing non-conventional variable
# names for simplicity
'''
This script reads all csv files in the input folder and merges them into a single dataframe.
The merged dataframe is then saved to the output folder.
The script also logs the ingestion details to a text file in the output folder.
'''

import os
from pathlib import Path
import json
from datetime import datetime
import logging

import pandas as pd

# Configure logging to output to the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Load config.json and get input and output paths
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

ENV = config["active_environment"]
input_folder_path = Path(os.getcwd(), config[ENV]["input_folder_path"])
output_folder_path = Path(os.getcwd(), config[ENV]["output_folder_path"])


def merge_multiple_data_sources():
    """Read all csv files in the input folder and merge them into a single dataframe."""
    df_list = []
    log_list = []

    for file_path in input_folder_path.iterdir():
        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path, low_memory=False)

            logger.info(
                "File to ingest: %s , (cols: %s, rows: %s)",
                file_path.name, df.shape[1], df.shape[0]
            )

            log_list.append({
                "ingest_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file": file_path.name,
                "source_location": str(input_folder_path),
                "row_count": len(df)
            })

            df_list.append(df)

    ingest_df = pd.concat(df_list, ignore_index=True)

    # save concatenated df to output folder
    ingest_df = ingest_df.drop_duplicates()
    logger.info(
        "Final data shape after merging: %s cols, %s rows.\n ",
        ingest_df.shape[1],
        ingest_df.shape[0])

    ingest_df.to_csv(Path(output_folder_path, "finaldata.csv"), index=False)
    logger.info(
        "Saved final data version to %s",
        Path(
            output_folder_path,
            "finaldata.csv"))

    # log record of ingested source files
    with open(Path(output_folder_path, "ingestedfiles.txt"), "a+", encoding="utf-8") as log_file:
        for log_entry in log_list:
            log_file.write(
                f"{log_entry['ingest_time']}, {log_entry['file']}, "
                f"{log_entry['source_location']}, {log_entry['row_count']}\n"
            )
            logger.info(
                "Logged ingestion details for file: %s", log_entry["file"]
            )


# This block ensures that the merge function is called only when this
# script is executed directly.
if __name__ == "__main__":
    merge_multiple_data_sources()
