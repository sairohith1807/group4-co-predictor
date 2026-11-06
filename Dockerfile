# Group 4 – Air Pollution Forecasting (CO(GT)) : MLflow + FastAPI + Streamlit in one image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    API_URL=http://127.0.0.1:8000

WORKDIR /app

# 1) dependencies (cached layer)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 2) train + log + register the model at build time, so visitors get a ready model
#    (uses data/AirQualityUCI.csv if you put it there, otherwise downloads it from UCI)
COPY data/ data/
COPY train.py .
ARG FINAL_MODEL=RF
ARG FINAL_IMPUTER=knn
RUN FINAL_MODEL=$FINAL_MODEL FINAL_IMPUTER=$FINAL_IMPUTER python train.py

# 3) the services (changing these files does not retrain the model)
COPY api.py app.py start.sh ./
RUN chmod +x start.sh \
 && useradd --create-home appuser \
 && chown -R appuser /app
USER appuser

EXPOSE 8501 8000 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["./start.sh"]
