from pydantic import BaseModel, Field, validator


class ForecastRequest(BaseModel):
    """Request schema for forecast generation."""

    state: str = Field(..., description="State name for forecast generation")
    forecast_periods: int = Field(8, description="Number of future periods to forecast")

    @validator("state")
    def state_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("State must be a non-empty string")
        return value.strip()

    @validator("forecast_periods")
    def forecast_periods_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("forecast_periods must be a positive integer")
        return value
