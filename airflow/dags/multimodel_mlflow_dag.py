import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd
from airflow.decorators import task
from airflow.models import DAG
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.dates import days_ago
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

NAME = "DecadenceFull"
BUCKET = "mlops-final"
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET = "MedHouseVal"
EXPERIMENT_NAME = NAME
DAG_ID = NAME

DATETIME_FMT = "%Y%m%d %H%M"

models = dict(
    zip(
        ["RandomForest", "LinearRegression", "HistGB"],
        [RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()],
    )
)

default_args = {
    "owner": NAME,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    dag_id=DAG_ID,
    description="Обучение моделей и трекинг с помощью MLFlow",
    start_date=days_ago(2),
    schedule="30 4 * * *",
    catchup=False,
    default_args=default_args,
)


@task
def init() -> Dict[str, Any]:
    metrics = {}
    metrics["start_timestamp"] = datetime.now().strftime(DATETIME_FMT)
    metrics["experiment_name"] = EXPERIMENT_NAME

    try:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=f"s3://{BUCKET}/{EXPERIMENT_NAME}"
        )
        mlflow.set_experiment(EXPERIMENT_NAME)
    except mlflow.exceptions.MlflowException as err:
        if "already exists" not in err.message:
            raise err

        experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

    metrics["experiment_id"] = experiment_id

    with mlflow.start_run(
        run_name="parent_run",
        experiment_id=experiment_id,
        description="Parent run for training in DAG",
    ) as parent_run:
        metrics["run_id"] = parent_run.info.run_id
        return metrics


@task
def get_data_from_postgres(metrics: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    metrics["data_download_start"] = datetime.now().strftime(DATETIME_FMT)

    pg_hook = PostgresHook("pg_connection")
    con = pg_hook.get_conn()
    data = pd.read_sql_query("select * from california_housing", con)

    s3_hook = S3Hook("s3_connection")
    file_name = f"{NAME}/datasets/california_housing.pkl"
    pickle_byte_obj = pickle.dumps(data)
    s3_hook.load_bytes(pickle_byte_obj, file_name, bucket_name=BUCKET, replace=True)

    metrics["data_download_end"] = datetime.now().strftime(DATETIME_FMT)
    return metrics


@task
def prepare_data(metrics: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    metrics["data_preparation_start"] = datetime.now().strftime(DATETIME_FMT)

    s3_hook = S3Hook("s3_connection")

    file_name = f"{NAME}/datasets/california_housing.pkl"
    data = s3_hook.download_file(key=file_name, bucket_name=BUCKET)
    data: pd.DataFrame = pd.read_pickle(data)

    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.transform(X_test)

    for data_name, data in zip(
        ["X_train", "X_test", "y_train", "y_test"],
        [X_train_fitted, X_test_fitted, y_train, y_test],
    ):
        pickle_byte_obj = pickle.dumps(data)
        s3_hook.load_bytes(
            pickle_byte_obj,
            f"{NAME}/datasets/{data_name}.pkl",
            bucket_name=BUCKET,
            replace=True,
        )

    metrics["data_preparation_end"] = datetime.now().strftime(DATETIME_FMT)
    return metrics


def train_mlflow_model(
    model: Any,
    name: str,
    X_train: np.array,
    X_test: np.array,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    signature = infer_signature(X_test, prediction)
    model_info = mlflow.sklearn.log_model(model, name, signature=signature)
    mlflow.evaluate(
        model_info.model_uri,
        data=X_test,
        targets=y_test.values,
        model_type="regressor",
        evaluators=["default"],
    )


@task
def train_model(metrics: Dict[str, Any], model_name: str, **kwargs) -> Dict[str, Any]:
    metrics[f"train_start_{model_name}"] = datetime.now().strftime(DATETIME_FMT)

    s3_hook = S3Hook("s3_connection")

    data = {}
    for data_name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(
            f"{NAME}/datasets/{data_name}.pkl", bucket_name=BUCKET
        )
        data[data_name] = pd.read_pickle(file)

    run_id = metrics["run_id"]
    experiment_id = metrics["experiment_id"]

    with mlflow.start_run(run_id=run_id) as parent_run:
        with mlflow.start_run(
            run_name=model_name, experiment_id=experiment_id, nested=True
        ) as child_run:
            train_mlflow_model(
                models[model_name],
                model_name,
                data["X_train"],
                data["X_test"],
                data["y_train"],
                data["y_test"],
            )

    metrics[f"train_end_{model_name}"] = datetime.now().strftime(DATETIME_FMT)
    return metrics


@task
def save_results(metrics: Dict[str, Any], **kwargs) -> None:
    result = {}

    date = datetime.now().strftime("%Y_%m_%d_%H")
    file_name = f"{NAME}/results/{date}.json"

    for m in metrics:
        result = {**result, **m}

    result["end_timestamp"] = datetime.now().strftime(DATETIME_FMT)

    s3_hook = S3Hook("s3_connection")
    json_byte_obj = json.dumps(result)
    s3_hook.load_string(json_byte_obj, file_name, bucket_name=BUCKET, replace=True)

    _LOG.info("Success.")


with dag:
    metrics = init()
    metrics = get_data_from_postgres(metrics)
    metrics = prepare_data(metrics)
    metrics = [
        train_model.override(task_id=f"train_model_{m}")(metrics, m)
        for m in models.keys()
    ]
    save_results(metrics)
