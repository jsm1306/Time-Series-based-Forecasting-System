from pydantic import BaseModel, Field
from typing import List


class ForecastResponse(BaseModel):
    """Response schema for forecast results."""

    state: str = Field(..., description="State used for forecast generation")
    model_used: str = Field(..., description="Model selected for prediction")
    forecast_periods: int = Field(..., description="Number of forecast periods returned")
    predictions: List[float] = Field(..., description="Forecasted values")
    generated_at: str = Field(..., description="ISO timestamp for forecast generation")
