from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.config.settings import get_logger
from app.schemas.request_schema import ForecastRequest
from app.schemas.response_schema import ForecastResponse
from app.services.prediction_service import PredictionService

router = APIRouter()
service = PredictionService()
logger = get_logger("forecast_routes")


@router.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint for the forecasting API."""
    return JSONResponse(status_code=200, content={"status": "healthy"})


@router.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest) -> ForecastResponse:
    """Generate a forecast for a requested state and horizon."""
    logger.info(
        "Received forecast request for state=%s periods=%d model=%s",
        request.state,
        request.forecast_periods,
        request.model_name,
    )
    try:
        prediction = service.predict(request.state, request.forecast_periods, request.model_name)
        return ForecastResponse(**prediction)
    except ValueError as error:
        logger.warning("Forecast request failed: %s", str(error))
        raise HTTPException(status_code=404, detail=str(error))
    except Exception:
        logger.exception("Forecast generation failed")
        raise HTTPException(status_code=500, detail="Forecast generation failed")
