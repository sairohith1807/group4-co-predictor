# Group 4 – Air Pollution Forecasting and Anomalous Sensor Readings

**Team:** Sai Rohith CHETLAPALLY · Srija POUDAPALLY
**Course:** Machine Learning – ESAIP Angers (M1 Cybersecurity & AI)
**Dataset:** [UCI Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality) (De Vito et al., 2008) – hourly readings from a low-cost gas sensor array in an Italian city, March 2004 – April 2005
**Target:** `CO(GT)` – true hourly carbon monoxide concentration (mg/m³)

---

## 1. Research question

How sensitive are predictions of an air pollutant to

- **(a)** the way we validate over time (chronological vs random split),
- **(b)** the way we handle missing values (median, causal forward-fill, KNN, native NaN), and
- **(c)** the removal of anomalous sensor readings (Isolation Forest) from the training data?

**Hypotheses (stated before training)**

- **H1** – A random split gives a lower (optimistic) error than a chronological split, because neighbouring hours are strongly autocorrelated.
- **H2** – Removing Isolation-Forest anomalies from the training set does not noticeably reduce the test error, because many unusual readings are real pollution episodes, not sensor faults.

The full analysis (data inspection, leakage check, supervised models, Isolation Forest, the three experiments, MLflow tracking and the API) is in
[`notebook/Group4_Air_Pollution_MLflow_API.ipynb`](notebook/Group4_Air_Pollution_MLflow_API.ipynb) (Google Colab: *Runtime → Run all*).

## 2. Method (summary)

| Item | Choice |
|---|---|
| Missing values | `-200` in the raw file → `NaN`; rows without a target are dropped (target is never imputed) |
| Leakage control | `PT08.S1(CO)` (sensor built to measure CO) and the other reference analysers (`NMHC(GT)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)`) are **excluded** |
| Features | `PT08.S2(NMHC)`, `PT08.S3(NOx)`, `PT08.S4(NO2)`, `PT08.S5(O3)`, `T`, `RH`, `AH` + `hour`, `dayofweek`, `month` |
| Protocol | Chronological hold-out: first 75 % of time = train, last 25 % = test; every preprocessing step fitted on train only (scikit-learn `Pipeline`) |
| Models | Random Forest, HistGradientBoosting; baselines: training mean, persistence (t-1) |
| Unsupervised | Isolation Forest (contamination 1 %, 3 %, 5 %) on the physical features only |
| Metrics | MAE, RMSE, R², median absolute error; 3 seeds (mean ± std) |
| Final deployed model | **Random Forest + KNN imputer** (best chronological RMSE in Experiment 2) |

## 3. MLOps / deployment

```
               docker build (train.py)                         docker run (start.sh)
UCI data ──► train + log runs ──► register model ──►  MLflow UI  :5000
             (MLflow, sqlite)     group4_co_gt_predictor   FastAPI   :8000  ──► loads model from the registry
                                                            Streamlit :8501  ──► calls the API
```

| Service | Port | What it does |
|---|---|---|
| **Streamlit app** | 8501 | Sliders for each sensor → predicted CO(GT); simulate a failed sensor; batch CSV predictions; predicted vs true chart on the test set; table of MLflow runs |
| **FastAPI** | 8000 | `GET /health`, `POST /predict`, `POST /predict_batch`; interactive Swagger page at `/docs` |
| **MLflow UI** | 5000 | Experiments, runs, metrics and the registered model |

The model is **trained, logged and registered in MLflow during `docker build`**, with fixed seeds and the same protocol as the notebook, so everyone who builds the image gets the same model.

## 4. Run the Docker image

You only need **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** (open it and wait for *Engine running*).

### Option A – pull the published image

```bash
docker run -p 8501:8501 -p 8000:8000 -p 5000:5000 ghcr.io/poudapallysrija1520-glitch/group4-air-pollution-co-predictor:latest
```

### Option B – build it yourself from this repository

```bash
git clone https://github.com/poudapallysrija1520-glitch/group4-air-pollution-co-predictor.git
cd group4-air-pollution-co-predictor
docker build -t group4-co-predictor:1.0 .
docker run -d --name group4-co -p 8501:8501 -p 8000:8000 -p 5000:5000 group4-co-predictor:1.0
```

On Windows you can also double-click `1_BUILD_IMAGE.bat`, then `2_RUN_PROJECT.bat`.

### Option C – image file (offline)

```bash
docker load -i group4-co-predictor.tar
docker run -p 8501:8501 -p 8000:8000 -p 5000:5000 group4-co-predictor:1.0
```

Then open:

- App → http://localhost:8501
- API docs → http://localhost:8000/docs
- MLflow → http://localhost:5000

Stop: `docker rm -f group4-co`

## 5. Try the API

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"timestamp":"2005-03-01T08:00:00","PT08_S2_NMHC":1046,"PT08_S3_NOx":1056,"PT08_S4_NO2":1692,"PT08_S5_O3":1268,"T":13.6,"RH":48.9,"AH":0.7578}'
```

Response: `{"timestamp": "2005-03-01T08:00:00", "CO_GT_pred_mg_m3": ...}`

- A missing sensor can be sent as `null` – the pipeline's imputer fills it.
- A reading with no sensor value at all is rejected with HTTP `422`.

## 6. Repository structure

| Path | Role |
|---|---|
| `Dockerfile` | Python 3.11-slim → install packages → run `train.py` → copy services → non-root user, healthcheck |
| `train.py` | Load data, compare 5 model/imputer candidates, log to MLflow, register the final model, write `artifacts/` |
| `api.py` | FastAPI service; loads `models:/group4_co_gt_predictor/latest` (fallback: joblib copy) |
| `app.py` | Streamlit front-end calling the API |
| `start.sh` | Starts MLflow UI, FastAPI and Streamlit inside the container |
| `requirements.txt` | Python dependencies |
| `docker-compose.yml` | `docker compose up --build` |
| `.github/workflows/docker-publish.yml` | Builds, smoke-tests and publishes the image to `ghcr.io` on every push |
| `*.bat` | Windows helpers: build, run, save for visitors, stop |
| `VISITOR_RUN.bat`, `VISITOR_README.txt` | Given to visitors together with the image `.tar` |
| `notebook/` | Full Colab analysis notebook |
| `data/` | Optional `AirQualityUCI.csv`; otherwise downloaded from UCI during the build |

## 7. Limitations

- One city, 13 months, one target pollutant; results may not transfer to other places or seasons.
- The chronological test block covers only winter–spring, so seasonal drift affects the numbers.
- Hyper-parameters were fixed (not heavily tuned) and only three seeds were used to estimate noise.
- Without fault labels, Isolation Forest flags can only be described, not validated as sensor failures.

## 8. Reference

S. De Vito, E. Massera, M. Piga, L. Martinotto, G. Di Francia (2008). *On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario.* Sensors and Actuators B: Chemical, 129(2), 750–757. Dataset: UCI Machine Learning Repository (CC BY 4.0).
