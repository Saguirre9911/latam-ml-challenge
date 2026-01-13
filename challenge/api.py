from functools import lru_cache
from pathlib import Path
from typing import List

import fastapi
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel

from challenge.model import DelayModel

app = fastapi.FastAPI()


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


class PredictRequest(BaseModel):
    flights: List[Flight]


@lru_cache(maxsize=1)
def _load_reference_values() -> dict:
    data_path = Path(__file__).resolve().parents[1] / "data" / "data.csv"
    df = pd.read_csv(data_path, usecols=["OPERA", "TIPOVUELO", "MES"])
    return {
        "opera": set(df["OPERA"].dropna().unique()),
        "tipo_vuelo": set(df["TIPOVUELO"].dropna().unique()),
        "mes": set(df["MES"].dropna().unique()),
    }


def _validate_flights(flights: List[Flight]) -> None:
    refs = _load_reference_values()
    for flight in flights:
        if flight.OPERA not in refs["opera"]:
            raise HTTPException(status_code=400, detail="Invalid OPERA value.")
        if flight.TIPOVUELO not in refs["tipo_vuelo"]:
            raise HTTPException(status_code=400, detail="Invalid TIPOVUELO value.")
        if flight.MES not in refs["mes"]:
            raise HTTPException(status_code=400, detail="Invalid MES value.")


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    _validate_flights(request.flights)
    data = pd.DataFrame([flight.model_dump() for flight in request.flights])
    model = DelayModel()
    features = model.preprocess(data)
    predictions = model.predict(features)
    return {"predict": predictions}
