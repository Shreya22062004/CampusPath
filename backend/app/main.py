from pathlib import Path
import json
import joblib
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import math

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "campus_placement_data.csv"
ARTIFACTS = ROOT / "backend" / "artifacts"

app = FastAPI(title="Student Placement Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=[origin.strip() for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",") if origin.strip()],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

class PredictionRequest(BaseModel):
    student_data: dict
    model_name: str = "RandomForest"
    feature_set: str = "All"

def load_metadata():
    path = ARTIFACTS / "metadata.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Models are not trained yet. Run: python -m backend.app.train_models")
    return json.loads(path.read_text(encoding="utf-8"))

@app.get("/api/health")
def health():
    return {"status": "ok", "models_trained": (ARTIFACTS / "metadata.json").exists()}

@app.get("/api/schema")
def schema():
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Dataset not found in data/")
    df = pd.read_csv(DATA_PATH, nrows=2000)
    excluded = {"placed", "salary_lpa", "student_id"}
    fields = []
    for col in df.columns:
        if col in excluded:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            vals = pd.to_numeric(series, errors="coerce").dropna()
            fields.append({"name": col, "kind": "number",
                           "min": float(vals.min()) if len(vals) else 0,
                           "max": float(vals.max()) if len(vals) else 100,
                           "default": float(vals.median()) if len(vals) else 0,
                           "required": True})
        else:
            options = sorted(series.dropna().astype(str).unique().tolist())
            if col == "specialization" and "No MBA" not in options:
                options.append("No MBA")
            # Keep dropdowns usable if the category list is large.
            fields.append({"name": col, "kind": "category", "options": options,
                           "default": options[0] if options else "", "required": True})
    return {"fields": fields}

@app.get("/api/models")
def models():
    meta = load_metadata()
    return {"models": meta["models"], "feature_sets": list(meta["feature_sets"].keys()),
            "feature_columns": meta["feature_sets"]}

@app.get("/api/feature-ranking")
def feature_ranking():
    # The artifacts CSV records feature order only. SHAP magnitudes are exported
    # separately by the training workflow; joining by feature prevents NaN UI values.
    path = ROOT / "backend" / "results" / "shap_feature_ranking.csv"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Feature ranking not generated yet.")
    ranking = pd.read_csv(path)
    if not {"feature", "mean_abs_shap"}.issubset(ranking.columns):
        raise HTTPException(status_code=503, detail="SHAP ranking file is missing required columns.")
    ranking["mean_abs_shap"] = pd.to_numeric(ranking["mean_abs_shap"], errors="coerce")
    ranking = ranking[ranking["feature"].notna() & ranking["mean_abs_shap"].map(math.isfinite)]
    if ranking.empty or (ranking["mean_abs_shap"] < 0).any():
        raise HTTPException(status_code=503, detail="SHAP ranking contains no valid non-negative values.")
    order_path = ARTIFACTS / "feature_ranking.csv"
    if order_path.exists():
        order = pd.read_csv(order_path)
        if {"feature", "rank"}.issubset(order.columns):
            ranking = order[["feature", "rank"]].merge(
                ranking.drop(columns=["rank"], errors="ignore"), on="feature", how="inner"
            )
    return ranking.sort_values("rank").to_dict(orient="records")

@app.get("/api/analytics")
def analytics():
    """Return held-out test metrics computed from the saved pipelines."""
    path = ROOT / "backend" / "results" / "saved_model_analytics.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Model analytics have not been generated. Run python -m backend.app.evaluate_saved_models")
    return json.loads(path.read_text(encoding="utf-8"))

@app.get("/api/shap-summary")
def shap_summary():
    path = ROOT / "backend" / "results" / "shap_summary_beeswarm.png"
    if not path.exists():
        raise HTTPException(status_code=503, detail="SHAP summary plot is unavailable. Run the training SHAP analysis first.")
    return FileResponse(path, media_type="image/png", filename="shap-summary.png")

@app.post("/api/predict")
def predict(request: PredictionRequest):
    meta = load_metadata()
    if request.model_name not in meta["models"]:
        raise HTTPException(status_code=400, detail="Unknown model name.")
    if request.feature_set not in meta["feature_sets"]:
        raise HTTPException(status_code=400, detail="Unknown feature set.")
    columns = meta["feature_sets"][request.feature_set]
    safe_model = request.model_name.lower()
    safe_set = request.feature_set.lower().replace("-", "_")
    model_path = ARTIFACTS / f"{safe_model}__{safe_set}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=503, detail="Requested model artifact is missing. Retrain models.")
    try:
        row = {col: request.student_data[col] for col in columns}
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Missing input field: {exc.args[0]}")
    source = pd.read_csv(DATA_PATH, nrows=2000)
    for col, value in row.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            raise HTTPException(status_code=422, detail=f"Missing input field: {col}")
        if pd.api.types.is_numeric_dtype(source[col]):
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise HTTPException(status_code=422, detail=f"{col} must be numeric.")
            if not math.isfinite(number):
                raise HTTPException(status_code=422, detail=f"{col} must be a finite number.")
            bounds = pd.to_numeric(source[col], errors="coerce").dropna()
            if len(bounds) and not float(bounds.min()) <= number <= float(bounds.max()):
                raise HTTPException(status_code=422, detail=f"{col} must be between {bounds.min()} and {bounds.max()}.")
            row[col] = number
        else:
            category = str(value)
            valid = set(source[col].dropna().astype(str).unique())
            if col == "specialization":
                valid.add("No MBA")
            if category not in valid:
                raise HTTPException(status_code=422, detail=f"Invalid value for {col}.")
            row[col] = category
    X = pd.DataFrame([row], columns=columns)
    pipeline = joblib.load(model_path)
    try:
        probabilities = pipeline.predict_proba(X)[0]
        class_index = list(pipeline.classes_).index(meta["positive_class"])
        probability = float(probabilities[class_index])
    except (ValueError, IndexError, AttributeError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not calculate placement probability: {exc}")
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise HTTPException(status_code=500, detail="Model returned an invalid placement probability.")
    prediction = int(probability >= 0.5)
    return {
        "prediction": "Likely to be placed" if prediction == 1 else "May need placement support",
        "predicted_class": prediction,
        "placement_probability": round(probability, 4),
        "model_name": request.model_name,
        "feature_set": request.feature_set,
        "features_used": columns,
        "note": "This is a model estimate from the supplied dataset, not a guarantee of an individual's employment outcome."
    }
