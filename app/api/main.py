from fastapi import APIRouter

from app.api.routes import stt, models

api_router = APIRouter()
api_router.include_router(stt.router)
api_router.include_router(models.router)