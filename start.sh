#!/bin/sh
# Starts the three services of the Group 4 container.
set -e
cd "$(dirname "$0")"

echo "Starting MLflow UI      -> http://localhost:5000"
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 > /tmp/mlflow.log 2>&1 &

echo "Starting FastAPI        -> http://localhost:8000/docs"
uvicorn api:app --host 0.0.0.0 --port 8000 --log-level warning &

echo "Starting Streamlit app  -> http://localhost:8501"
exec streamlit run app.py \
  --server.port 8501 --server.address 0.0.0.0 --server.headless true \
  --browser.gatherUsageStats false
