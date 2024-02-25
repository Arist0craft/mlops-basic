# MLOps
Проект для курса по начальному MLOps. В проекте используются Apache Airflow и MLFlow. 
Вся инфраструктура построена на Docker Compose

## Настройка проекта

Для корректной работы необходимо создать папку .aws и файлы

__config__:
```
[default]
region = #Указать регион
```

__credentials__:
```
[default]
aws_access_key_id = # Указать ID ключа 
aws_secret_access_key = # Указать сам ключ
endpoint_url = https://storage.yandexcloud.net

[airflow]
endpoint_url = https://storage.yandexcloud.net
```

Также в корне проекта необходимо создать файл с переменными окружения __.env__:
```
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=airflow
POSTGRES_PORT=5432
# AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://postgres:postgres@postgres:5432/airflow

AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:${POSTGRES_PORT}/${POSTGRES_DB}
AIRFLOW__WEBSERVER__SECRET_KEY="test_secret_key" # Можно указать любой ключ
AIRFLOW__WEBSERVER__DEFAULT_UI_TIMEZONE="Europe/Moscow"
AIRFLOW__CORE__DEFAULT_TIMEZONE="Europe/Moscow"

MLFLOW_DB=mlflow_db
MLFLOW_S3_ARTIFACT_ROOT= # Укажите корневой путь к бакету S3, где будут храниться артефакты
MLFLOW_S3_ENDPOINT_URL="https://storage.yandexcloud.net"
MLFLOW_TRACKING_URI="postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5434/${MLFLOW_DB}"

PYTHONDONTWRITEBYTECODE=1
```

Любые названия и креды для баз данных можно поменять на свои при желании