# Group 4 – Air Pollution Forecasting (CO(GT)) – Docker image

**Team:** Sai Rohith CHETLAPALLY · Srija POUDAPALLY
**Dataset:** UCI Air Quality (De Vito et al., 2008) · **Target:** `CO(GT)` (mg/m³)

One image contains the whole MLOps pipeline:

| Service | Port | What visitors see |
|---|---|---|
| Streamlit app | **8501** | Sliders → prediction, batch CSV, predicted vs true chart, MLflow runs |
| FastAPI | **8000** | REST API, interactive Swagger page at `/docs` |
| MLflow UI | **5000** | Experiments, runs, metrics, registered model `group4_co_gt_predictor` |

The model is **trained, logged and registered in MLflow during `docker build`** (chronological 75/25 split, `PT08.S1(CO)` and reference analysers excluded, RF + KNN imputer), so visitors get a ready model.

## For visitors – run it (only Docker Desktop needed)

```bash
docker run -p 8501:8501 -p 8000:8000 -p 5000:5000 ghcr.io/<github-user>/<repo>:latest
```
Open http://localhost:8501 (app) · http://localhost:8000/docs (API) · http://localhost:5000 (MLflow).

## How the image is built

Every push to `main` triggers **GitHub Actions** (`.github/workflows/docker-publish.yml`):
build → train the model inside the image → smoke-test `/health`, `/predict` and Streamlit → publish to `ghcr.io`.
After the first build, set the package to **Public** (GitHub → your profile → Packages → the package → Package settings → Change visibility).

Build locally instead (needs Docker Desktop):
```bash
docker build -t group4-co-predictor:1.0 .
docker run -p 8501:8501 -p 8000:8000 -p 5000:5000 group4-co-predictor:1.0
```
Share as a file: `docker save group4-co-predictor:1.0 -o group4-co-predictor.tar` → visitors `docker load -i group4-co-predictor.tar`.

## Test the API

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"timestamp":"2005-03-01T08:00:00","PT08_S2_NMHC":1046,"PT08_S3_NOx":1056,"PT08_S4_NO2":1692,"PT08_S5_O3":1268,"T":13.6,"RH":48.9,"AH":0.7578}'
```
Missing sensors can be `null` (the imputer fills them); a reading with no sensor at all returns `422`.

## Files

| File | Role |
|---|---|
| `Dockerfile` | Installs deps → runs `train.py` → copies services → non-root user, healthcheck |
| `train.py` | Loads data, compares 5 candidates, logs to MLflow, registers the final model, writes `artifacts/` |
| `api.py` | FastAPI; loads `models:/group4_co_gt_predictor/latest` (fallback: joblib copy) |
| `app.py` | Streamlit front-end calling the API |
| `start.sh` | Starts MLflow UI, FastAPI and Streamlit |
| `docker-compose.yml` | `docker compose up --build` |
| `.github/workflows/docker-publish.yml` | Cloud build + test + publish |
| `data/AirQualityUCI.csv` | Dataset (CC BY 4.0, UCI); downloaded at build time if missing |
