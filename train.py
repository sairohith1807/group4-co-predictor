"""
Group 4 - Air Pollution Forecasting (UCI Air Quality, target CO(GT))
Runs once during `docker build`:
  1. loads the UCI data (data/AirQualityUCI.csv if present, otherwise downloads it),
  2. applies the same protocol as the notebook (chronological 75/25 split, no leakage),
  3. compares candidate models and logs everything to MLflow (sqlite:///mlflow.db),
  4. trains, logs and REGISTERS the final model (default RF + KNN imputer),
  5. writes the files the API and the Streamlit app need into artifacts/.
"""
import io
import json
import os
import urllib.request
import warnings
import zipfile

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ------------------------------- CONFIG -------------------------------
SEED = 42
TARGET = "CO(GT)"
PROXY_COLS = ["PT08.S1(CO)"]          # direct proxy of the target -> excluded (leakage)
TRAIN_FRAC = 0.75                     # chronological split
FINAL_MODEL = os.environ.get("FINAL_MODEL", "RF")
FINAL_IMPUTER = os.environ.get("FINAL_IMPUTER", "knn")
MODEL_NAME = "group4_co_gt_predictor"
EXPERIMENT = "Group4_AirQuality_CO"
UCI_URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
ART = "artifacts"
# ----------------------------------------------------------------------

# JSON field name (API) -> training column name
COL_MAP = {"PT08_S2_NMHC": "PT08.S2(NMHC)", "PT08_S3_NOx": "PT08.S3(NOx)",
           "PT08_S4_NO2": "PT08.S4(NO2)", "PT08_S5_O3": "PT08.S5(O3)",
           "T": "T", "RH": "RH", "AH": "AH"}
LABELS = {"PT08_S2_NMHC": "PT08.S2 (NMHC sensor)", "PT08_S3_NOx": "PT08.S3 (NOx sensor)",
          "PT08_S4_NO2": "PT08.S4 (NO2 sensor)", "PT08_S5_O3": "PT08.S5 (O3 sensor)",
          "T": "Temperature T (°C)", "RH": "Relative humidity RH (%)", "AH": "Absolute humidity AH"}

SKOPS_TRUSTED = [
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.ensemble._hist_gradient_boosting.binning._BinMapper",
    "sklearn._loss.link.IdentityLink", "sklearn._loss.link.Interval",
    "sklearn._loss.loss.HalfSquaredError", "sklearn._loss._loss.CyHalfSquaredError",
    "numpy.dtype",
]


def load_raw():
    local = os.path.join("data", "AirQualityUCI.csv")
    if os.path.exists(local):
        print("Loaded local file:", local)
        return pd.read_csv(local, sep=";", decimal=",")
    print("Downloading dataset from UCI ...")
    data = urllib.request.urlopen(UCI_URL, timeout=120).read()
    z = zipfile.ZipFile(io.BytesIO(data))
    name = [n for n in z.namelist() if n.endswith("AirQualityUCI.csv")][0]
    return pd.read_csv(z.open(name), sep=";", decimal=",")


def prepare(raw):
    raw = raw.dropna(axis=1, how="all").dropna(axis=0, how="all")
    raw["datetime"] = pd.to_datetime(raw["Date"].astype(str) + " " + raw["Time"].astype(str),
                                     format="%d/%m/%Y %H.%M.%S")
    df = raw.drop(columns=["Date", "Time"]).set_index("datetime").sort_index()
    df = df[~df.index.duplicated()]
    return df.apply(pd.to_numeric, errors="coerce").replace(-200, np.nan)   # -200 = missing


def make_model(name, seed):
    if name == "RF":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    if name == "HGB":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=seed)
    raise ValueError(name)


def make_pipeline(name, imputer="median", seed=SEED):
    steps = []
    if imputer == "median":
        steps.append(("imputer", SimpleImputer(strategy="median")))
    elif imputer == "knn":
        steps += [("scaler", StandardScaler()), ("imputer", KNNImputer(n_neighbors=5))]
    elif imputer == "none":
        assert name == "HGB", "only HistGradientBoosting accepts NaN natively"
    steps.append(("model", make_model(name, seed)))
    return Pipeline(steps)


def metrics(y_true, y_pred):
    return dict(MAE=float(mean_absolute_error(y_true, y_pred)),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
                R2=float(r2_score(y_true, y_pred)),
                MedAE=float(median_absolute_error(y_true, y_pred)))


def log_sklearn_model(pipe, X_test, pred):
    kwargs = dict(signature=infer_signature(X_test, pred), input_example=X_test.dropna().head(3),
                  registered_model_name=MODEL_NAME)
    try:
        mlflow.sklearn.log_model(pipe, name="model", skops_trusted_types=SKOPS_TRUSTED, **kwargs)
    except TypeError:            # MLflow version without skops options
        mlflow.sklearn.log_model(pipe, name="model", **kwargs)


def main():
    np.random.seed(SEED)
    os.makedirs(ART, exist_ok=True)

    df = prepare(load_raw())
    gt_others = [c for c in df.columns if c.endswith("(GT)") and c != TARGET]
    X_all = df.drop(columns=gt_others + [TARGET] + PROXY_COLS)
    X_all["hour"] = X_all.index.hour
    X_all["dayofweek"] = X_all.index.dayofweek
    X_all["month"] = X_all.index.month
    y_all = df[TARGET]

    valid = y_all.notna()
    X_v, y_v = X_all[valid], y_all[valid]
    n_train = int(len(X_v) * TRAIN_FRAC)
    X_train, X_test = X_v.iloc[:n_train], X_v.iloc[n_train:]
    y_train, y_test = y_v.iloc[:n_train], y_v.iloc[n_train:]
    features = list(X_train.columns)
    print(f"Train {len(X_train)} rows, test {len(X_test)} rows, features: {features}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)
    config = dict(seed=SEED, target=TARGET, train_frac=TRAIN_FRAC, excluded_proxy=str(PROXY_COLS),
                  n_train=len(X_train), n_test=len(X_test))

    # ---- model selection (chronological hold-out), one nested run per candidate ----
    with mlflow.start_run(run_name="model_selection"):
        mlflow.log_params(config)
        for m, imp in [("RF", "median"), ("HGB", "median"), ("RF", "knn"), ("HGB", "knn"), ("HGB", "none")]:
            with mlflow.start_run(run_name=f"model={m} | imputer={imp}", nested=True):
                p = make_pipeline(m, imp, SEED).fit(X_train, y_train)
                r = metrics(y_test, p.predict(X_test))
                mlflow.log_params({"model": m, "imputer": imp})
                mlflow.log_metrics(r)
                print(f"  {m:3s} + {imp:6s}: RMSE={r['RMSE']:.3f}  MAE={r['MAE']:.3f}  R2={r['R2']:.3f}")

    # ---- final model: train, log, register ----
    with mlflow.start_run(run_name=f"final_{FINAL_MODEL}_{FINAL_IMPUTER}") as run:
        pipe = make_pipeline(FINAL_MODEL, FINAL_IMPUTER, SEED).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        final_metrics = metrics(y_test, pred)
        mlflow.log_params({**config, "model": FINAL_MODEL, "imputer": FINAL_IMPUTER, "features": ",".join(features)})
        mlflow.log_metrics(final_metrics)
        try:
            log_sklearn_model(pipe, X_test, pred)
        except Exception as e:   # the API falls back to the joblib copy below
            print("WARNING: MLflow model registration failed:", str(e)[:300])
        run_id = run.info.run_id
    joblib.dump(pipe, os.path.join(ART, "model.joblib"))
    print("Final model:", {k: round(v, 3) for k, v in final_metrics.items()}, "| run id:", run_id)

    # ---- files for the API and the Streamlit app ----
    fields = {f: {"col": c, "label": LABELS[f],
                  "min": float(X_train[c].quantile(0.01)), "max": float(X_train[c].quantile(0.99)),
                  "median": float(X_train[c].median())} for f, c in COL_MAP.items()}
    meta = dict(model_name=MODEL_NAME, run_id=run_id, features=features, col_map=COL_MAP, fields=fields,
                test_metrics={k: round(v, 3) for k, v in final_metrics.items()},
                model_label=f"{FINAL_MODEL} + {FINAL_IMPUTER} imputer",
                default_ts=pd.Timestamp(X_test.index[0]).isoformat(),
                target_median=float(y_train.median()))
    with open(os.path.join(ART, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    sample = X_test.head(24 * 14)
    out = pd.DataFrame({"timestamp": [pd.Timestamp(t).isoformat() for t in sample.index]})
    for f, c in COL_MAP.items():
        out[f] = sample[c].values
    out["true_CO_GT"] = y_test.loc[sample.index].values
    out.to_csv(os.path.join(ART, "api_test_sample.csv"), index=False)

    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
    keep = [c for c in ["tags.mlflow.runName", "metrics.RMSE", "metrics.MAE", "metrics.R2", "start_time"] if c in runs]
    runs[keep].rename(columns=lambda c: c.split(".")[-1]).to_csv(os.path.join(ART, "mlflow_runs.csv"), index=False)
    print("Artifacts written to", ART)


if __name__ == "__main__":
    main()
