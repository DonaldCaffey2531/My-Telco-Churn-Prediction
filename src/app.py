from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from keras import load_model

import tensorflow as tf

# Load scaler & model
scaler = joblib.load("scaler.pkl")
model = tf.keras.models.load_model("../models/churn_model")

app = FastAPI()

class CustomerData(BaseModel):
    features: list  # Example: [tenure, MonthlyCharges, ..., InternetService_FiberOptic]

@app.post("/predict")
def predict(data: CustomerData):
    # Convert input
    X = np.array(data.features).reshape(1, -1)
    X = scaler.transform(X)  # Apply same scaling as training

    # Prediction
    prob = model.predict(X)[0][0]
    churn = int(prob > 0.5)

    return {"churn_probability": float(prob), "churn_prediction": churn}
