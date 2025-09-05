import os
import logging
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from src.api.pydantic_models import PreprocessedRequest, RawTransactionRequest, PredictionResponse

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fraud_api")

# --- Load environment variables ---
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")  # path inside container

# --- FastAPI app ---
app = FastAPI(
    title="Fraud Detection API",
    description="Predict fraud from transaction data using a locally bundled model",
    version="1.0"
)

# --- Global model variable ---
model = None

# --- Startup event ---
@app.on_event("startup")
def load_model():
    global model
    try:
        logger.info(f"📦 Loading model from local path: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        logger.info("✅ Model loaded successfully.")
    except Exception as e:
        logger.exception(f"❌ Failed to load model: {e}")
        model = None

# --- Health check ---
@app.get("/health")
def health():
    if model is None:
        return {"status": "error", "detail": "Model not loaded"}
    return {"status": "ok", "model_path": MODEL_PATH}

# --- Predict from preprocessed features ---
@app.post("/predict-preprocessed", response_model=PredictionResponse)
def predict_preprocessed(req: PreprocessedRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    X = np.array(req.features, dtype=float).reshape(1, -1)
    y_pred = model.predict(X)[0]
    try:
        proba = model.predict_proba(X)[0, 1]
    except Exception:
        proba = None

    return PredictionResponse(prediction=int(y_pred), probability=proba)

# --- Predict from raw transaction data ---
@app.post("/predict-raw", response_model=PredictionResponse)
def predict_raw(req: RawTransactionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    X_df = pd.DataFrame([req.dict()])
    y_pred = model.predict(X_df)[0]
    try:
        proba = model.predict_proba(X_df)[0, 1]
    except Exception:
        proba = None

    return PredictionResponse(prediction=int(y_pred), probability=proba)
