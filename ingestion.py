"""
This script reads all CSV files in the input folder and merges them into a single dataframe.
The merged dataframe is then saved to the output folder.
The script also logs the ingestion details to a text file in the output folder.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

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


def merge_multiple_data_sources(files_to_ingest: Optional[List[str]] = None) -> None:
    """
    Reads all CSV files in the input folder (or specified files) and merges them into a single dataframe.
    The merged dataframe is saved to the output folder, and ingestion details are logged.

    Args:
        files_to_ingest (Optional[List[str]]): List of specific file names to ingest. If None, all CSV files
                                               in the input folder are ingested.

    Returns:
        None
    """
    df_list: List[pd.DataFrame] = []
    log_list: List[Dict[str, Any]] = []

    # Determine files to ingest
    if files_to_ingest is None:
        files_to_ingest = [file.name for file in input_folder_path.iterdir() if file.suffix == ".csv"]

    # Process each file
    for file_name in files_to_ingest:
        file_path = Path(input_folder_path, file_name)
        if file_path.exists() and file_path.suffix == ".csv":
            try:
                # Read the CSV file
                df = pd.read_csv(file_path, low_memory=False)
                logger.info(
                    "File to ingest: %s , (cols: %s, rows: %s)",
                    file_path.name,
                    df.shape[1],
                    df.shape[0],
                )

                # Log ingestion details
                log_list.append(
                    {
                        "ingest_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "file": file_path.name,
                        "source_location": str(input_folder_path),
                        "row_count": len(df),
                    }
                )

                # Append dataframe to the list
                df_list.append(df)
            except Exception as e:
                logger.error("Error reading file %s: %s", file_name, str(e))
        else:
            logger.warning("File %s does not exist or is not a CSV file.", file_name)

    # Merge all dataframes into one
    if df_list:
        try:
            ingest_df = pd.concat(df_list, ignore_index=True).drop_duplicates()
            logger.info(
                "Final data shape after merging: %s cols, %s rows.",
                ingest_df.shape[1],
                ingest_df.shape[0],
            )

            # Save the merged dataframe to the output folder
            output_file_path = Path(output_folder_path, "finaldata.csv")
            ingest_df.to_csv(output_file_path, index=False)
            logger.info("Saved final data version to %s", output_file_path)

            # Log record of ingested source files
            log_file_path = Path(output_folder_path, "ingestedfiles.txt")
            with open(log_file_path, "a+", encoding="utf-8") as log_file:
                for log_entry in log_list:
                    log_file.write(
                        f"{log_entry['ingest_time']}, {log_entry['file']}, "
                        f"{log_entry['source_location']}, {log_entry['row_count']}\n"
                    )
                    logger.info("Logged ingestion details for file: %s", log_entry["file"])
        except Exception as e:
            logger.error("Error during merging or saving: %s", str(e))
    else:
        logger.warning("No files were ingested. Dataframe list is empty.")


if __name__ == "__main__":
    """
    This block ensures that the merge function is called only when this script is executed directly.
    """
    merge_multiple_data_sources()
