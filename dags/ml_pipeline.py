import logging
from typing import Any, Dict

import pandas as pd
from airflow.decorators import dag, task
from pendulum import datetime
from sklearn.model_selection import train_test_split

from steps import deploy_model, hyperparameter_tuning, register_model, train_model

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


@dag(
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
)
def mlflow_tutorial_dag():
    @task
    def build_dataset_stage():
        # Read the wine-quality csv file from the URL
        csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
        try:
            data = pd.read_csv(csv_url, sep=";")
        except Exception as e:
            logger.exception(
                "Unable to download training & test CSV, check your internet connection. Error: %s",
                e,
            )

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)
        train, valid = train_test_split(train)

        # The predicted column is "quality" which is a scalar from [3, 9]
        train_x = train.drop(["quality"], axis=1)
        val_x = valid.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)
        train_y = train[["quality"]]
        val_y = valid[["quality"]]
        test_y = test[["quality"]]
        return {
            "train_x": train_x,
            "val_x": val_x,
            "test_x": test_x,
            "train_y": train_y,
            "val_y": val_y,
            "test_y": test_y,
        }

    @task
    def hyperparameter_tuning_stage(dataset: Dict[str, Any]):
        return hyperparameter_tuning(dataset, 10, "rmse", "tpe.suggest")

    # @task
    # def train_model_stage(dataset: Dict[str, Any]):
    #     alpha = l1_ratio = 0.5
    #     train_model(alpha, l1_ratio, dataset)

    @task
    def register_model_stage(mlflow_run_id: str):
        return register_model(mlflow_run_id)

    @task
    def deploy_model_stage(mlflow_run_id: str):
        return deploy_model(mlflow_run_id)

    deploy_model_stage(register_model_stage(hyperparameter_tuning_stage(build_dataset_stage())))


mlflow_tutorial_dag()
