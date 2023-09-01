import logging
from typing import Any, Dict, List

from airflow.decorators import dag, task
from datetime import datetime 

from model_experimentation import (
    deploy_model,
    hyperparameter_tuning,
    register_model,
    trigger_model_reload,
)
from dataset_gen import check_for_new_data, prepare_new_data

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


@dag(
    dag_id="mlops_pipeline",
    start_date=datetime(2014, 1, 1),
    schedule="@yearly",
    fail_stop=True,
    max_active_runs=1
)
def mlflow_tutorial_dag():
    @task.short_circuit()
    def check_for_new_data_stage(prev_ds=None, ds=None):
        prev_date = datetime.strptime(prev_ds, "%Y-%m-%d")
        current_date = datetime.strptime(ds, "%Y-%m-%d")
        return check_for_new_data(prev_date, current_date)
    
    @task
    def build_dataset_stage(csv_urls: List[str]):
        train_x, val_x, test_x, train_y, val_y, test_y = prepare_new_data(csv_urls)
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
        deploy_model(mlflow_run_id)
        return trigger_model_reload()

    deploy_model_stage(register_model_stage(hyperparameter_tuning_stage(build_dataset_stage(check_for_new_data_stage()))))


mlflow_tutorial_dag()
