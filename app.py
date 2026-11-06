"""Streamlit front-end for visitors: calls the FastAPI service running in the same container."""
import json
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

API = os.environ.get("API_URL", "http://127.0.0.1:8000")
HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
st.set_page_config(page_title="Group 4 – CO(GT) predictor", page_icon="🌫️", layout="wide")


@st.cache_data
def load_meta():
    with open(os.path.join(ART, "meta.json")) as f:
        return json.load(f)


meta = load_meta()
FIELDS = meta["fields"]


def api_ok():
    try:
        return requests.get(f"{API}/health", timeout=3).json()
    except Exception:
        return None


st.title("🌫️ Group 4 – Air quality: CO(GT) predictor")
st.caption("Sai Rohith CHETLAPALLY · Srija POUDAPALLY — UCI Air Quality · MLflow · FastAPI · Streamlit · Docker")

health = api_ok()
if health is None:
    st.warning("The API is still starting (it loads the model). Wait a few seconds and refresh this page.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
m = health["test_metrics"]
c1.metric("Test MAE (mg/m³)", f'{m["MAE"]:.3f}')
c2.metric("Test RMSE (mg/m³)", f'{m["RMSE"]:.3f}')
c3.metric("Test R²", f'{m["R2"]:.3f}')
c4.metric("Model", meta["model_label"])

tab1, tab2, tab3 = st.tabs(["Single reading", "Batch (CSV)", "Test set & MLflow"])

with tab1:
    st.subheader("Predict CO(GT) for one hourly reading")
    left, right = st.columns([2, 1])
    with left:
        d = st.date_input("Date", value=pd.Timestamp(meta["default_ts"]).date())
        t = st.time_input("Hour", value=pd.Timestamp(meta["default_ts"]).time(), step=3600)
        values = {}
        cols = st.columns(2)
        for i, (field, info) in enumerate(FIELDS.items()):
            with cols[i % 2]:
                missing = st.checkbox(f"{info['label']} missing", key=f"miss_{field}")
                values[field] = None if missing else st.slider(
                    info["label"], float(info["min"]), float(info["max"]), float(info["median"]), key=field)
    with right:
        st.write("")
        if st.button("Predict", type="primary", width="stretch"):
            payload = {"timestamp": pd.Timestamp.combine(d, t).isoformat(), **values}
            r = requests.post(f"{API}/predict", json=payload, timeout=30)
            if r.ok:
                y = r.json()["CO_GT_pred_mg_m3"]
                st.metric("Predicted CO(GT)", f"{y:.2f} mg/m³")
                level = "low" if y < 2 else "moderate" if y < 4 else "high"
                st.write(f"Level: **{level}** (training median: {meta['target_median']:.2f} mg/m³)")
            else:
                st.error(r.json().get("detail", r.text))
            with st.expander("Request sent to the API"):
                st.json(payload)
        st.info("Tick “missing” to simulate a failed sensor – the model's imputer fills it.")

with tab2:
    st.subheader("Predict many readings from a CSV")
    st.write("Columns: `timestamp` plus any of " + ", ".join(f"`{f}`" for f in FIELDS))
    template = pd.read_csv(os.path.join(ART, "api_test_sample.csv")).drop(columns=["true_CO_GT"]).head(24)
    st.download_button("Download example CSV (24 test hours)", template.to_csv(index=False),
                       "example_readings.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type="csv")
    df = pd.read_csv(up) if up is not None else None
    if df is None and st.checkbox("Use the example file"):
        df = template
    if df is not None:
        rows = [{k: (None if pd.isna(v) else v) for k, v in r.items() if k in FIELDS or k == "timestamp"}
                for r in df.to_dict("records")]
        r = requests.post(f"{API}/predict_batch", json=rows, timeout=60)
        if r.ok:
            out = df.copy()
            out["CO_GT_pred_mg_m3"] = [p["CO_GT_pred_mg_m3"] for p in r.json()]
            st.line_chart(out.set_index(pd.to_datetime(out["timestamp"]))["CO_GT_pred_mg_m3"])
            st.dataframe(out, width="stretch")
            st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv", "text/csv")
        else:
            st.error(r.text)

with tab3:
    st.subheader("API predictions vs true CO(GT) on the chronological test set")
    test = pd.read_csv(os.path.join(ART, "api_test_sample.csv"))
    n = st.slider("Hours to show", 24, len(test), min(168, len(test)), step=24)
    sub = test.head(n)
    payload = [{k: (None if pd.isna(v) else v) for k, v in r.items() if k != "true_CO_GT"}
               for r in sub.to_dict("records")]
    preds = [p["CO_GT_pred_mg_m3"] for p in requests.post(f"{API}/predict_batch", json=payload, timeout=60).json()]
    chart = pd.DataFrame({"true CO(GT)": sub["true_CO_GT"].values, "predicted": preds},
                         index=pd.to_datetime(sub["timestamp"]))
    st.line_chart(chart)
    err = np.abs(chart["predicted"] - chart["true CO(GT)"])
    st.write(f"MAE on these {n} hours: **{err.mean():.3f} mg/m³**")

    st.subheader("MLflow runs")
    runs_csv = os.path.join(ART, "mlflow_runs.csv")
    if os.path.exists(runs_csv):
        st.dataframe(pd.read_csv(runs_csv), width="stretch")
    st.caption(f"Registered model: `{health['model']}` · run id `{health['run_id']}` · source: {health['source']} "
               "· full MLflow dashboard on port 5000")
