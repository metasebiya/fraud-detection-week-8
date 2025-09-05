# src/api/pydantic_models.py

from pydantic import BaseModel, Field, conlist
from typing import Optional, Literal

class PreprocessedRequest(BaseModel):
    features: conlist(float, min_length=1) = Field(
        ..., description="List of preprocessed numerical features in the correct order"
    )

class RawTransactionRequest(BaseModel):
    purchase_value: float
    age: int
    source: Literal["SEO", "Ads", "Direct"]
    browser: Literal["Chrome", "Firefox", "Safari", "Edge"]
    sex: Literal["M", "F"]
    hour_of_day: int
    day_of_week: int
    country: str
    # Add other raw fields as needed

class PredictionResponse(BaseModel):
    prediction: int
    probability: Optional[float] = None
