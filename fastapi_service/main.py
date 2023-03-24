from fastapi import FastAPI
from typing import List

from fastapi_service.predictor import Predictor
from fastapi_service.transaction import Transaction


predictor = Predictor()
app = FastAPI()


# Запрос для одной транзакции
@app.post("/predict_one_transaction")
def predict_item(item: Transaction) -> float:
    return predictor.predict(item)[0]


# Запрос для списка транзакций
@app.post("/predict_transactions")
def predict_items(items: List[Transaction]) -> List[float]:
    return predictor.predict(items).tolist()
