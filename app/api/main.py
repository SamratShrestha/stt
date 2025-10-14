from fastapi import APIRouter

from app.api.routes import stt

api_router = APIRouter()
api_router.include_router(stt.router)