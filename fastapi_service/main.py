from fastapi import FastAPI
from typing import List
from typing import Dict
from starlette.responses import JSONResponse

from fastapi_service.celery_config.celery_utils import get_task_info
from fastapi_service.celery_tasks import tasks
from fastapi_service.db_connection import DBConnection
from fastapi_service.metrics_utils import PrometheusMiddleware, metrics
from fastapi_service.transaction import Transaction


def create_app() -> FastAPI:
    current_app = FastAPI()
    current_app.add_middleware(PrometheusMiddleware)
    current_app.add_route("/metrics", metrics)
    return current_app


app = create_app()
db = None

@app.on_event('startup')
async def startup_event():
    global db
    db = DBConnection()

# Запрос для одной транзакции
@app.post("/predict_one_transaction")
async def predict_item(item: Transaction):
    task = tasks.get_prediction_one_task.apply_async(args=[item])
    return JSONResponse({"task_id": task.id})

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
async def predict_items(items: List[Transaction]):
    task = tasks.get_prediction_list_task.apply_async(args=[items])
    return JSONResponse({"task_id": task.id})

# Запрос для обучения модели
@app.post("/train_input_data")
async def train_model_on_input_data(data: Dict):
    task = tasks.get_train_model_task.apply_async(args=[data])
    return JSONResponse({"task_id": task.id})

@app.post("/train_full_data")
async def train_model_on_full_data(data: Dict):
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

    task = tasks.get_train_model_task.apply_async(args=[data])
    return JSONResponse({"task_id": task.id})

@app.get("/task/{task_id}")
async def get_task_status(task_id: str) -> dict:
    return get_task_info(task_id)

# Бросить исключение (тестируем панель Total Exceptions)
@app.get("/raise_exception")
def raise_exception():
    raise Exception
