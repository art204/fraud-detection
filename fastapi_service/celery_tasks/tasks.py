from celery import shared_task

from fastapi_service.celery_config.celery_utils import create_celery
from fastapi_service.predictor import Predictor
from fastapi_service.settings import Settings


celery = create_celery()
settings = Settings()
predictor = Predictor(settings.get_model_name())


@shared_task(bind=True,autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3},
             name='celery_queue:predict_one')
def get_prediction_one_task(self, item):
    return predictor.predict(item).tolist()

@shared_task(bind=True,autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3},
             name='celery_queue:predict_list')
def get_prediction_list_task(self, items):
    return predictor.predict(items).tolist()

@shared_task(bind=True,autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3},
             name='celery_queue:train_model')
def get_train_model_task(self, data):
    return predictor.train(data)
