#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
from hydra.utils import get_original_cwd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# DO NOT MODIFY
def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)
    
    # Drop outliers
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    
    # Save the cleaned file
    df.to_csv('clean_sample.csv', index=False)

    # Log the new data.
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

# Fill in the data type and description for each argument
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")
  
    parser.add_argument(
        "--input_artifact", 
        type=str,  # Data type is string
        help="The input artifact containing the raw data.",  # Description
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,  # Data type is string
        help="The name of the output artifact for the cleaned data.",  # Description
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,  # Data type is string
        help="The type of the output artifact (e.g., cleaned_data).",  # Description
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,  # Data type is string
        help="A description of the output artifact.",  # Description
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,  # Data type is float
        help="The minimum price for filtering the data.",  # Description
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,  # Data type is float
        help="The maximum price for filtering the data.",  # Description
        required=True
    )

    args = parser.parse_args()
    go(args)
