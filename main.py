from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf
import os

app = FastAPI()

print("📦 Loading model & scaler...")
MODEL_PATH = "btc_gru_model.h5"
SCALER_PATH = "scaler.pkl"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

# Request body model
class SequenceIn(BaseModel):
    sequence: list[float]  # 24 prices

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
def predict(input: SequenceIn):
    sequence = input.sequence

    if len(sequence) != 24:
        return {"error": "Sequence must have 24 prices"}

    entry_price = sequence[-1]

    # Scale & reshape
    sequence_scaled = scaler.transform(np.array(sequence).reshape(-1, 1)).reshape(1, 24, 1)

    # Predict
    pred_scaled = model.predict(sequence_scaled)[0][0]
    pred = scaler.inverse_transform(np.array([[0, pred_scaled]]))[0][1]

    reco = get_recommendation(pred, entry_price)

    return {
        "entry_price": float(entry_price),
        "predicted_price": float(pred),
        "recommendation": reco
    }
