from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from binance.client import Client
import numpy as np
import joblib
import tensorflow as tf
import os
import logging

# 🔷 Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

# 🔷 Load model & scaler
MODEL_PATH = "btc_gru_model.h5"
SCALER_PATH = "scaler.pkl"

logging.info("📦 Loading model & scaler...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    logging.error(f"❌ Failed to load model or scaler: {e}")
    raise RuntimeError("Failed to load model or scaler. Check paths and files.")

# 🔷 Setup Binance client
api_key = os.getenv("BINANCE_API_KEY", "")
api_secret = os.getenv("BINANCE_API_SECRET", "")
if not api_key or not api_secret:
    logging.warning("⚠️ BINANCE_API_KEY or BINANCE_API_SECRET not set!")

client = Client(api_key, api_secret)


def fetch_last_24_prices():
    try:
        logging.info("📈 Fetching last 24h prices from Binance...")
        klines = client.get_historical_klines(
            "BTCUSDT",
            Client.KLINE_INTERVAL_1HOUR,
            "24 hours ago UTC",
            "now"
        )
        closes = [float(k[4]) for k in klines[-24:]]
        logging.info(f"✅ Fetched {len(closes)} closing prices.")
        return closes
    except Exception as e:
        logging.error(f"❌ Error fetching data from Binance: {e}")
        raise RuntimeError(f"Error fetching data from Binance: {e}")


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
def predict():
    try:
        # Fetch prices
        sequence = fetch_last_24_prices()
        if len(sequence) < 24:
            raise ValueError("Insufficient data: less than 24 hourly prices.")

        entry_price = sequence[-1]
        logging.info(f"📌 Entry price: {entry_price}")

        # Scale & reshape
        sequence_scaled = scaler.transform(np.array(sequence).reshape(-1, 1)).reshape(1, 24, 1)

        # Predict
        pred_scaled = model.predict(sequence_scaled)[0][0]
        pred = scaler.inverse_transform(np.array([[0, pred_scaled]]))[0][1]

        logging.info(f"🤖 Predicted price: {pred}")

        # Get recommendation
        reco = get_recommendation(pred, entry_price)
        logging.info(f"📝 Recommendation: {reco}")

        return {
            "entry_price": float(entry_price),
            "predicted_price": float(pred),
            "recommendation": reco
        }

    except Exception as e:
        logging.error(f"❌ Prediction failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )
