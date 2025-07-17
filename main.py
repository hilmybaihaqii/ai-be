from fastapi import FastAPI, Request
from pydantic import BaseModel
from binance.client import Client
import numpy as np
import joblib
import tensorflow as tf
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

logger.info("📦 Loading model & scaler...")
MODEL_PATH = "btc_gru_model.h5"
SCALER_PATH = "scaler.pkl"

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    logger.exception("❌ Failed to load model or scaler: %s", e)
    raise

api_key = os.getenv("BINANCE_API_KEY", "")
api_secret = os.getenv("BINANCE_API_SECRET", "")
client = Client(api_key, api_secret)

class SequenceRequest(BaseModel):
    sequence: list[float] | None = None  # Optional, fallback ke Binance kalau None

def fetch_last_24_prices():
    try:
        klines = client.get_historical_klines(
            "BTCUSDT",
            Client.KLINE_INTERVAL_1HOUR,
            "24 hours ago UTC",
            "now"
        )
        closes = [float(k[4]) for k in klines[-24:]]
        if len(closes) != 24:
            raise ValueError("Fetched data from Binance is not 24 points.")
        return closes
    except Exception as e:
        logger.error("❌ Failed to fetch from Binance: %s", e)
        raise

def get_recommendation(pred, entry):
    if pred > entry * 1.01:
        return "Strong BUY"
    elif pred > entry:
        return "BUY"
    elif pred < entry * 0.99:
        return "Strong SELL"
    else:
        return "SELL"

@app.get("/")
def root():
    return {"message": "✅ AI Backend is running"}

@app.post("/predict")
def predict(req: SequenceRequest):
    if req.sequence:
        sequence = req.sequence
        logger.info("🚀 Received sequence from client.")
    else:
        logger.info("🌐 Fetching sequence from Binance.")
        sequence = fetch_last_24_prices()

    if len(sequence) != 24:
        return {"error": f"❌ Sequence must have exactly 24 prices, got {len(sequence)}"}

    entry_price = sequence[-1]

    try:
        # Scale & reshape
        sequence_scaled = scaler.transform(np.array(sequence).reshape(-1, 1)).reshape(1, 24, 1)
        pred_scaled = model.predict(sequence_scaled)[0][0]
        pred = scaler.inverse_transform(np.array([[0, pred_scaled]]))[0][1]
    except Exception as e:
        logger.error("❌ Prediction error: %s", e)
        return {"error": "Prediction error"}

    reco = get_recommendation(pred, entry_price)

    logger.info("✅ Prediction done: Entry: %.2f, Pred: %.2f, Reco: %s", entry_price, pred, reco)

    return {
        "entry_price": float(entry_price),
        "predicted_price": float(pred),
        "recommendation": reco
    }
