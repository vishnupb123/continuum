from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.health import router as health_router
from app.api.journals import router as journals_router
from app.core.config import settings

from app.api.auth import router as auth_router

app = FastAPI(title="Continuum API", version="0.0.1-m0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router)
app.include_router(journals_router)
app.include_router(auth_router)
