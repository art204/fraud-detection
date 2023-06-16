from fastapi import FastAPI
from typing import List
from typing import Dict

from fastapi_service.metrics_utils import PrometheusMiddleware, metrics
from fastapi_service.predictor import Predictor
from fastapi_service.settings import Settings
from fastapi_service.transaction import Transaction


settings = Settings()
predictor = Predictor(settings.get_model_name())
app = FastAPI()
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics)

# Запрос для одной транзакции
@app.post("/predict_one_transaction")
def predict_item(item: Transaction) -> float:
    return predictor.predict(item)[0]


# Запрос для списка транзакций
@app.post("/predict_transactions")
def predict_items(items: List[Transaction]) -> List[float]:
    return predictor.predict(items).tolist()

# Запрос для обучения модели
@app.post("/train_model")
def predict_items(data: Dict) -> bool:
    return predictor.train(data)

# Бросить исключение (тестируем панель Total Exceptions)
@app.get("/raise_exception")
def predict_items():
    raise Exception
