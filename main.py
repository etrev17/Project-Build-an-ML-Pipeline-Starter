import json
import mlflow
import tempfile
import os
import wandb
import hydra
import yaml
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]

# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            # Run the basic_cleaning step
            mlflow.run(
                "src/basic_cleaning",
                entry_point="main",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "cleaned_data",
                    "min_price": config["basic_cleaning"]["min_price"],
                    "max_price": config["basic_cleaning"]["max_price"]
                },
            )

        if "data_check" in active_steps:
            # Run the data_check step
            mlflow.run(
                "src/data_check",
                entry_point="main",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["data_check"]["min_price"],
                    "max_price": config["data_check"]["max_price"]
                },
            )

        if "data_split" in active_steps:
            # Implement data_split step
            pass

        if "train_random_forest" in active_steps:

            # Serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # Run the train_random_forest step
            mlflow.run(
                "src/train_random_forest",
                entry_point="main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "rf_config": rf_config,
                    "output_artifact": "random_forest_export",
                    "output_type": "model_export",
                    "output_description": "Trained Random Forest model"
                },
            )

        if "test_regression_model" in active_steps:
            # Implement test_regression_model step
            pass


if __name__ == "__main__":
    go()
