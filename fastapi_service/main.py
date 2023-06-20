from fastapi import FastAPI
from typing import List
from typing import Dict

from fastapi_service.db_connection import DBConnection
from fastapi_service.metrics_utils import PrometheusMiddleware, metrics
from fastapi_service.predictor import Predictor
from fastapi_service.settings import Settings
from fastapi_service.transaction import Transaction


app = FastAPI()
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics)
settings = None
predictor = None
db = None

@app.on_event('startup')
async def startup_event():
    global settings
    global predictor
    global db
    settings = Settings()
    predictor = Predictor(settings.get_model_name())
    db = DBConnection()

# Запрос для одной транзакции
@app.post("/predict_one_transaction")
def predict_item(item: Transaction) -> float:
    return predictor.predict(item)[0]

@app.post("/add_transaction")
async def add_item(item: Transaction) -> dict:
    trn = await db.get(item.TransactionID)
    if trn is None:
        await db.put(item)
        return {'id': item.TransactionID, 'status': 'added'}
    else:
        return {'id': item.TransactionID, 'status': 'already in db'}


# Запрос для списка транзакций
@app.post("/predict_transactions")
def predict_items(items: List[Transaction]) -> List[float]:
    return predictor.predict(items).tolist()

# Запрос для обучения модели
@app.post("/train_input_data")
def train_model_on_input_data(data: Dict) -> bool:
    return predictor.train(data)

@app.post("/train_full_data")
async def train_model_on_full_data(data: Dict) -> bool:
    items_list = await db.get_all()
    ids_set = set()
    for item in items_list:
        ids_set.add(item.TransactionID)
    for i in range(len(data['TransactionID'])):
        item_id = data['TransactionID'][i]
        if item_id not in ids_set:
            d = {}
            for k in data:
                d[k] = data[k][i]
            item = Transaction.parse_obj(d)
            items_list.append(item)
            await db.put(item)
    return predictor.train(items_list)

# Бросить исключение (тестируем панель Total Exceptions)
@app.get("/raise_exception")
def raise_exception():
    raise Exception
