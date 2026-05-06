from fastapi import FastAPI
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from app.api.routes.forecast_routes import router as forecast_router

app = FastAPI(
    title="Forecasting API",
    description="Production-ready forecasting API for state-level predictions.",
    version="1.0.0",
)

app.include_router(forecast_router)
