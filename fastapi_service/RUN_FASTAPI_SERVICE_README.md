## Project source files
Project source files are stored in folder `fastapi_service/`.

Prometheus and Grafana are used for monitoring. Dashboard file (fastapi-observability.json) and
utils classes and methods (file metrics_utils.py) are taken from https://github.com/blueswen/fastapi-observability.

## Start service
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
and `Percent  of 5xx Requests` grafana panels.

## Dashboard
Check predefined dashboard `FastAPI Service` on Grafana [http://localhost:3000/](http://localhost:3000/)
The dashboard is available on [Grafana Dashboards](https://grafana.com/grafana/dashboards/16110)