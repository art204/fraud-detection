## Project source files
Project source files are stored in folder `fastapi_service/`.

## Docker image link
Docker image of the service in docker hub: 
https://hub.docker.com/r/artemd2204/fraud_detection_fastapi_service

Run docker container with command:
```commandline
docker run --rm -it artemd2204/fraud_detection_fastapi_service
```

## Start service with monitoring
Prometheus and Grafana are used for monitoring. Dashboard file (`fastapi-observability.json`) and
utils classes and methods (file `metrics_utils.py`) are taken from https://github.com/blueswen/fastapi-observability.

Start fastapi service, prometheus and grafana:
```commandline
cd fastapi_service
docker compose up --build
```

## Endpoints:

`http://127.0.0.1:8000/predict_one_transaction` - prediction for one transaction. Use `POST` method with `"Content-Type": "application/json"`
and one transaction in json format in request body.

`http://127.0.0.1:8000/predict_transactions` - prediction for list of transactions. Use `POST` method with `"Content-Type": "application/json"`
and list of transactions in json format in request body.

`http://127.0.0.1:8000/raise_exception` - use request with `GET` method for testing `Total Exception`
and `Percent of 5xx Requests` grafana panels.

## Test
Service test done in `service_test/fastapi_service_test.ipynb`

## Dashboard
Check predefined dashboard `FastAPI Service` on Grafana [http://localhost:3000/](http://localhost:3000/)
The dashboard is available on [Grafana Dashboards](https://grafana.com/grafana/dashboards/16110)
