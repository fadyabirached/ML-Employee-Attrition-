"""FastAPI service wrapping the employee attrition model.

Same model + pipeline as app.py, exposed as a JSON API instead of a
Streamlit form -- for programmatic/integration use rather than a human
clicking through a UI.

Run locally:
    uvicorn api:app --reload

Then either open http://127.0.0.1:8000/docs for interactive docs, or:
    curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
        "satisfaction_level": 0.38, "last_evaluation_rating": 0.53,
        "projects_worked_on": 2, "average_monthly_hours": 157,
        "time_spend_company": 3, "Work_accident": 0,
        "promotion_last_5years": 0, "Department": "sales", "salary": "low"
    }'
"""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from model_service import load_or_train_model, predict_one


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_or_train_model()
    yield


app = FastAPI(
    title="Employee Attrition Predictor",
    description="Predicts an employee's probability of leaving from HR features.",
    lifespan=lifespan,
)


class EmployeeFeatures(BaseModel):
    """Bounds mirror the training data's observed ranges.

    Values outside them are rejected rather than silently extrapolated.
    In-range values that combine into a profile the data never contains
    are still answered, but flagged via ``out_of_distribution``.
    """

    satisfaction_level: float = Field(ge=0.9, le=10)
    last_evaluation_rating: float = Field(ge=3.6, le=10)
    projects_worked_on: int = Field(ge=2, le=8)
    average_monthly_hours: int = Field(ge=96, le=320)
    time_spend_company: int = Field(ge=2, le=10)
    Work_accident: Literal[0, 1]
    promotion_last_5years: Literal[0, 1]
    Department: str
    salary: Literal["low", "medium", "high"]


class PredictionResponse(BaseModel):
    probability_leave: float
    will_leave: bool
    risk_band: Literal["low", "elevated", "high"]
    base_rate: float
    out_of_distribution: bool
    novelty_score: float
    latency_ms: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: EmployeeFeatures, request: Request) -> dict:
    return predict_one(request.app.state.bundle, features.model_dump())
