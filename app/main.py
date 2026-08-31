#Serve the ML model via FastAPI


import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"


mlflow.set_tracking_uri("sqlite:///mlflow.db")
MODEL_URI = "models:/aml-lgbm-v1/2"
model = mlflow.lightgbm.load_model(MODEL_URI)



FEATURE_ORDER = [
    "Amount Received", "Amount Paid",
    "Hour", "Day of week",
    "Cross currency check", "Cross bank check",
    "received_count_1h", "received_amt_1h",
    "received_count_24h", "received_amt_24h",
    "Receiving Currency", "Payment Currency", "Payment Format",
]
CAT_COLS = ["Receiving Currency", "Payment Currency", "Payment Format"]

app = FastAPI(title="AML detection")


#valid inputs into our model
class TransactionFeatures(BaseModel):
    amount_received: float = Field(alias="Amount Received")
    amount_paid: float = Field(alias="Amount Paid")
    hour: int = Field(alias="Hour", ge=0, le=23)
    day_of_week: int = Field(alias="Day of week", ge=0, le=6)
    cross_currency: int = Field(alias="Cross currency check", ge=0, le=1)
    cross_bank: int = Field(alias="Cross bank check", ge=0, le=1)
    received_count_1h: int = Field(alias="received_count_1h", ge=1)
    received_amt_1h: float = Field(alias="received_amt_1h")
    received_count_24h: int = Field(alias="received_count_24h", ge=1)
    received_amt_24h: float = Field(alias="received_amt_24h")
    receiving_currency: str = Field(alias="Receiving Currency")
    payment_currency: str = Field(alias="Payment Currency")
    payment_format: str = Field(alias="Payment Format")
    model_config = {"populate_by_name": True}


@app.get("/")
def frontend():
    return FileResponse(STATIC_DIR / "index.html")

    
@app.get("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/predict")
def predict(txn: TransactionFeatures, threshold: float = 0.5):
    row = pd.DataFrame([txn.model_dump(by_alias=True)])[FEATURE_ORDER]
    saved = model.booster_.pandas_categorical  # one list of levels per cat col
    for col, levels in zip(CAT_COLS, saved):
        row[col] = pd.Categorical(row[col], categories=levels).codes
    proba = float(model.predict_proba(row.to_numpy())[0, 1])
    return {
        "p_laundering": proba,
        "flag": proba >= threshold,
        "threshold": threshold,
    }