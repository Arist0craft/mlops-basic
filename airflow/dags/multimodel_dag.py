import json
import logging
import pickle
from datetime import datetime
from typing import Any, Dict, Literal

import pandas as pd
from airflow.decorators import task
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.dates import days_ago
from botocore.client import Config
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from airflow import DAG

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

BUCKET = "mlops"
DATA_PATH = "datasets/california_housing.pkl"

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

DATA_NAMES = ["X_train", "X_test", "y_train", "y_test"]

DEFAULT_ARGS = {
    "owner": "dimnktn",
    "retries": None,
}

TIME_FMT = "%Y%m%d %H:%M"


s3_client_cfg = Config(
    region_name="ru-central1",
    connect_timeout=600,
    read_timeout=600,
)

models = dict(
    zip(
        ["rf", "lr", "hgb"],
        [RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()],
    )
)


def create_dag(
    dag_id: str,
    model_name: Literal["rf", "lr", "hgb"],
):
    with DAG(
        dag_id,
        description="Первый пайплайн",
        schedule="0 1 * * *",
        start_date=days_ago(2),
        catchup=False,
        tags=["first"],
        default_args=DEFAULT_ARGS,
    ) as dag:

        @task()
        def init(**kwargs) -> Dict[str, Any]:
            metrics = {}
            metrics["model"] = model_name
            metrics["start_timestamp"] = datetime.now().strftime(TIME_FMT)

            logger.info("Train pipeline started.")
            return metrics

        @task()
        def get_data_from_postgres(metrics: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            # pg_hook = PostgresHook("pg_connection")
            # con = pg_hook.get_conn()
            # data = pd.read_sql_query("select * from california_housing", con)

            # s3_hook = S3Hook("s3_connection")
            # s3_client = s3_hook.get_client_type(config=s3_client_cfg)

            # pickle_obj = pickle.dumps(data)
            # s3_client.put_object(Body=pickle_obj, Bucket=BUCKET, Key=DATA_PATH)

            # logger.info("Data download finished")
            return metrics

        @task()
        def prepare_data(metrics: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            # s3_hook = S3Hook("s3_connection")

            # data = s3_hook.download_file(key=DATA_PATH, bucket_name=BUCKET)
            # data = pd.read_pickle(data)
            # X, y = data[FEATURES], data[TARGET]
            # X_train, X_test, y_train, y_test = train_test_split(
            #     X, y, test_size=0.2, random_state=42
            # )

            # scaler = StandardScaler()
            # X_train_fitted = scaler.fit_transform(X_train)
            # X_test_fitted = scaler.transform(X_test)

            # s3_client = s3_hook.get_client_type(config=s3_client_cfg)

            # for name, data in zip(
            #     DATA_NAMES,
            #     [X_train_fitted, X_test_fitted, y_train, y_test],
            # ):
            #     pickle_byte_obj = pickle.dumps(data)
            #     s3_client.put_object(
            #         Body=pickle_byte_obj, Bucket=BUCKET, Key=f"dataset/{name}.pkl"
            #     )

            # logger.info("Data preparation complete")
            return metrics

        @task()
        def train_model(metrics: Dict[str, Any], **kwargs) -> Dict[str, float]:
            s3_hook = S3Hook("s3_connection")

            data = {}
            for name in DATA_NAMES:
                file = s3_hook.download_file(f"dataset/{name}.pkl", bucket_name=BUCKET)
                data[name] = pd.read_pickle(file)

            model = models[metrics["model"]]
            metrics["train_start"] = datetime.now().strftime(TIME_FMT)
            model.fit(data["X_train"], data["y_train"])
            metrics["train_end"] = datetime.now().strftime(TIME_FMT)
            prediction = model.predict(data["X_test"])

            metrics["r2_score"] = r2_score(data["y_test"], prediction)
            metrics["rmse"] = mean_squared_error(data["y_test"], prediction) ** 0.5
            metrics["mse"] = median_absolute_error(data["y_test"], prediction)
            logger.info("Model training finished")

            return metrics

        @task()
        def save_results(metrics: Dict[str, Any], **kwargs) -> None:
            metrics["end_timestamp"] = datetime.now().strftime(TIME_FMT)

            s3_hook = S3Hook("s3_connection")

            date = datetime.now().strftime("%Y_%m_%d_%H")

            s3_client = s3_hook.get_client_type(config=s3_client_cfg)

            json_byte_obj = json.dumps(metrics)

            s3_client.put_object(
                Body=json_byte_obj,
                Bucket=BUCKET,
                Key=f"results/{metrics['model']}_{date}.json",
            )
            logger.info("Success.")

        metrics = init()
        metrics = get_data_from_postgres(metrics)
        metrics = prepare_data(metrics)
        metrics = train_model(metrics)
        save_results(metrics)


for model_name in models.keys():
    create_dag(f"{model_name}_train", model_name)
