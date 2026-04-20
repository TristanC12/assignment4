from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "taxi_tip_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
FEATURE_COLUMNS = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "pickup_hour",
    "pickup_day_of_week",
    "is_weekend",
    "trip_duration_minutes",
    "trip_speed_mph",
    "log_trip_distance",
    "fare_per_mile",
    "fare_per_minute",
    "pickup_borough",
    "dropoff_borough",
]

model_bundle: dict[str, Any] = {"model": None, "metadata": None, "loaded": False}


def load_model_files() -> tuple[Any, dict[str, Any]]:
    # load saved model
    model = joblib.load(MODEL_PATH)

    # load model info
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    else:
        metadata = {
            "model_name": "taxi-tip-regressor",
            "model_version": "local-1",
            "feature_names": FEATURE_COLUMNS,
            "training_metrics": {},
        }
    return model, metadata


def load_now():
    model, metadata = load_model_files()
    model_bundle["model"] = model
    model_bundle["metadata"] = metadata
    model_bundle["loaded"] = True


class TripFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passenger_count: float = Field(ge=0, le=8)
    trip_distance: float = Field(gt=0, le=100)
    fare_amount: float = Field(gt=0, le=500)
    pickup_hour: int = Field(ge=0, le=23)
    pickup_day_of_week: int = Field(ge=0, le=6)
    is_weekend: int = Field(ge=0, le=1)
    trip_duration_minutes: float = Field(gt=0, le=300)
    trip_speed_mph: float = Field(ge=0, le=100)
    log_trip_distance: float = Field(ge=0, le=10)
    fare_per_mile: float = Field(ge=0, le=300)
    fare_per_minute: float = Field(ge=0, le=50)
    pickup_borough: str = Field(min_length=1, max_length=40)
    dropoff_borough: str = Field(min_length=1, max_length=40)

    @field_validator("pickup_borough", "dropoff_borough")
    @classmethod
    def clean_text(cls, value: str) -> str:
        # trim text
        return value.strip()


class BatchRequest(BaseModel):
    records: list[TripFeatures] = Field(min_length=1, max_length=100)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_now()
    except Exception as exc:
        model_bundle["loaded"] = False
        model_bundle["startup_error"] = str(exc)
    yield


app = FastAPI(title="Taxi Tip Prediction API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "Something went wrong while handling the request.",
        },
    )


def predict_one(payload: TripFeatures):
    if model_bundle["model"] is None:
        load_now()

    # build one row frame
    row = {key: getattr(payload, key) for key in FEATURE_COLUMNS}
    X = pd.DataFrame([row])
    pred = float(model_bundle["model"].predict(X)[0])
    return max(0.0, pred)


@app.get("/health")
def health():
    metadata = model_bundle["metadata"] or {}
    return {
        "status": "ok" if model_bundle["loaded"] else "error",
        "model_loaded": model_bundle["loaded"],
        "model_version": metadata.get("model_version"),
    }


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    metadata = model_bundle["metadata"] or {}
    return {
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version"),
        "feature_names": metadata.get("feature_names", FEATURE_COLUMNS),
        "training_metrics": metadata.get("training_metrics", {}),
    }


@app.post("/predict")
def predict(payload: TripFeatures):
    
    metadata = model_bundle["metadata"] or {}
    prediction = predict_one(payload)
    return {
        "prediction_id": str(uuid.uuid4()),
        "predicted_tip_amount": round(prediction, 2),
        "model_version": metadata.get("model_version", "local-1"),
    }


@app.post("/predict/batch")
def predict_batch(payload: BatchRequest):
    metadata = model_bundle["metadata"] or {}
    predictions = [round(predict_one(record), 2) for record in payload.records]
    return {
        "count": len(predictions),
        "model_version": metadata.get("model_version", "local-1"),
        "predictions": predictions,
    }


try:
    load_now()
except Exception:
    pass
