# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import sys
import warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging

import click
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from dags.steps import hyperparameter_tuning

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


@click.command(
    help="Perform hyperparameter search with Hyperopt library. Optimize dl_train target."
)
@click.option(
    "--max-runs", type=click.INT, default=10, help="Maximum number of runs to evaluate."
)
@click.option(
    "--metric", type=click.STRING, default="rmse", help="Metric to optimize on."
)
@click.option(
    "--algo", type=click.STRING, default="tpe.suggest", help="Optimizer algorithm."
)
@click.option(
    "--seed", type=click.INT, default=97531, help="Seed for the random generator"
)
def search(max_runs, metric, algo, seed):
    warnings.filterwarnings("ignore")
    np.random.seed(seed)
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

    dataset = {
        "train_x": train_x,
        "val_x": val_x,
        "test_x": test_x,
        "train_y": train_y,
        "val_y": val_y,
        "test_y": test_y,
    }

    hyperparameter_tuning(dataset, max_runs, metric, algo)


if __name__ == "__main__":
    search()
