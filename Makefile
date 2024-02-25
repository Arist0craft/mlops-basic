build:
	docker build . -t mlops-c:latest

run:
	docker run \
		-it \
		-v ./.aws/:/home/airflow/.aws/ \
		--rm \
		--name mlops \
		mlops-c:latest \
		bash

stop:
	docker stop mlops

up:
	docker-compose up

down:
	docker-compose down

connect-airflow:
	docker exec -it airflow_webserver bash 

connect-mlflow:
	docker exec -it mlflow bash 