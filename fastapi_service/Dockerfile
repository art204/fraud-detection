FROM ubuntu

RUN apt-get update

RUN apt-get install -y python3 python3-pip

RUN mkdir /fastapi_service
ADD /fastapi_service/main.py /fastapi_service/main.py
ADD /fastapi_service/predictor.py /fastapi_service/predictor.py
ADD /fastapi_service/transaction.py /fastapi_service/transaction.py
ADD /fastapi_service/metrics_utils.py /fastapi_service/metrics_utils.py

RUN mkdir /anomaly
ADD /anomaly/feature_selection.py /anomaly/feature_selection.py
ADD /anomaly/models.py /anomaly/models.py

ADD /requirements.txt /requirements.txt

RUN pip install -r requirements.txt

CMD ["uvicorn", "--host", "0.0.0.0", "fastapi_service.main:app"]
