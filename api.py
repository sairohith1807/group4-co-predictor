"""FastAPI service: predicts hourly CO(GT) from the low-cost sensor array.
The model is loaded from the MLflow Model Registry (fallback: the joblib copy saved at build time)."""
import json
import os
from datetime import datetime
from typing import List, Optional

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "artifacts", "meta.json")) as f:
    META = json.load(f)
FEATURES, COL_MAP, MODEL_NAME = META["features"], META["col_map"], META["model_name"]

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(HERE, 'mlflow.db')}")
try:
    served_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
    MODEL_SOURCE = f"MLflow registry: models:/{MODEL_NAME}/latest"
except Exception as e:
    print("Registry load failed, using joblib copy:", str(e)[:200])
    served_model = joblib.load(os.path.join(HERE, "artifacts", "model.joblib"))
    MODEL_SOURCE = f"joblib copy of run {META['run_id']}"
print("Serving:", MODEL_SOURCE)


class Reading(BaseModel):
    timestamp: datetime
    PT08_S2_NMHC: Optional[float] = None
    PT08_S3_NOx: Optional[float] = None
    PT08_S4_NO2: Optional[float] = None
    PT08_S5_O3: Optional[float] = None
    T: Optional[float] = None
    RH: Optional[float] = None
    AH: Optional[float] = None
    model_config = {"json_schema_extra": {"example": {
        "timestamp": "2005-03-01T08:00:00", "PT08_S2_NMHC": 1046.0, "PT08_S3_NOx": 1056.0,
        "PT08_S4_NO2": 1692.0, "PT08_S5_O3": 1268.0, "T": 13.6, "RH": 48.9, "AH": 0.7578}}}


def to_frame(readings):
    rows = []
    for r in readings:
        d = {col: (np.nan if getattr(r, f) is None else getattr(r, f)) for f, col in COL_MAP.items()}
        d.update(hour=r.timestamp.hour, dayofweek=r.timestamp.weekday(), month=r.timestamp.month)
        rows.append(d)
    X = pd.DataFrame(rows)[FEATURES]
    sensors = [c for c in COL_MAP.values() if c in FEATURES]
    if X[sensors].isna().all(axis=1).any():
        raise HTTPException(status_code=422, detail="Each reading needs at least one sensor value.")
    return X


app = FastAPI(title="Group 4 – CO(GT) predictor", version="1.0",
              description="Predicts hourly CO(GT) in mg/m³ from the low-cost sensor array (UCI Air Quality). "
                          "Model tracked and registered with MLflow.")


@app.get("/")
def root():
    return {"message": "Group 4 CO(GT) API – open /docs to try it",
            "endpoints": ["/health", "/predict", "/predict_batch"]}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "source": MODEL_SOURCE, "run_id": META["run_id"],
            "test_metrics": META["test_metrics"], "features": FEATURES}


@app.post("/predict")
def predict(reading: Reading):
    y = served_model.predict(to_frame([reading]))[0]
    return {"timestamp": reading.timestamp, "CO_GT_pred_mg_m3": round(float(y), 3)}


@app.post("/predict_batch")
def predict_batch(readings: List[Reading]):
    if not readings:
        raise HTTPException(status_code=422, detail="Send at least one reading.")
    ys = served_model.predict(to_frame(readings))
    return [{"timestamp": r.timestamp, "CO_GT_pred_mg_m3": round(float(y), 3)} for r, y in zip(readings, ys)]
