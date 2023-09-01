from typing import Any, Dict
from catboost import CatBoostRegressor, Pool

import mlflow
import mlflow.sklearn
import numpy as np
from hyperopt import fmin, hp, rand, tpe
from mlflow.models import infer_signature
from mlflow.tracking.client import MlflowClient
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_NAME = "CatBoostModel"
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def hyperparameter_tuning(dataset, max_runs, metric, algo):
    def eval_fn():
        def eval(params):
            run_id = train_model(params, dataset)
            client = MlflowClient()
            training_run = client.get_run(run_id)
            metrics = training_run.data.metrics
            # return validation loss which will be used by the optimization algorithm
            valid_loss = metrics["val_{}".format(metric)]
            return valid_loss

        return eval

    # define the search space for hyper-parameters
    space = {
        "learning_rate": hp.uniform("learning_rate", 0.03, 0.1),
        "depth": hp.randint("depth", 4, 8),
        "l2_leaf_reg": hp.uniform("l2_leaf_reg", 0.5, 4)
    }
    with mlflow.start_run() as run:
        exp_id = run.info.experiment_id
        # run the optimization algorithm
        best = fmin(
            fn=eval_fn(),
            space=space,
            algo=tpe.suggest if algo == "tpe.suggest" else rand.suggest,
            max_evals=max_runs,
        )
        mlflow.set_tag("best params", str(best))
        # find all runs generated by this search
        client = MlflowClient()
        query = "tags.mlflow.parentRunId = '{run_id}' ".format(run_id=run.info.run_id)
        best_run = client.search_runs([exp_id], query, order_by=[f"metrics.val_{metric} ASC"])[0]
        mlflow.set_tag("best_run", best_run.info.run_id)
        return best_run.info.run_id


def train_model(params: Dict[str, Any], dataset: Dict[str, Any]) -> str:
    train_x = dataset["train_x"]
    train_y = dataset["train_y"]
    val_x = dataset["val_x"]
    val_y = dataset["val_y"]
    test_x = dataset["test_x"]
    test_y = dataset["test_y"]
    categorical_features = ["town", "flat_type", "storey_range", "flat_model"]
    train_pool = Pool(train_x, train_y,
                      cat_features=categorical_features)
    val_pool = Pool(val_x, val_y,
                    cat_features=categorical_features)
    test_pool = Pool(test_x, test_y,
                    cat_features=categorical_features)
    with mlflow.start_run(nested=True) as run:
        model = CatBoostRegressor(**params, iterations=500)
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        model.fit(train_pool, eval_set=val_pool)

        pred_y = model.predict(train_pool)
        rmse, mae, r2 = eval_metrics(train_y, pred_y)
        mlflow.log_metric("train_rmse", rmse)
        mlflow.log_metric("train_r2", r2)
        mlflow.log_metric("train_mae", mae)

        pred_y = model.predict(val_pool)
        rmse, mae, r2 = eval_metrics(val_y, pred_y)
        mlflow.log_metric("val_rmse", rmse)
        mlflow.log_metric("val_r2", r2)
        mlflow.log_metric("val_mae", mae)

        pred_y = model.predict(test_pool)
        rmse, mae, r2 = eval_metrics(test_y, pred_y)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("test_mae", mae)

        signature = infer_signature(test_x, pred_y)
        mlflow.sklearn.log_model(model, "model", signature=signature)

    return run.info.run_id


def register_model(mlflow_run_id: str) -> str:
    model_version = mlflow.register_model(f"runs:/{mlflow_run_id}/model",MODEL_NAME)
    return model_version.version


def deploy_model(model_version: str):
    client = MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version,
        stage="Production",
        archive_existing_versions = True
    )

def trigger_model_reload():
    response = requests.post("http://api:8081")
    while response.status_code != 200:
        response = requests.post("http://api:8081")
