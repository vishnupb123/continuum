from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from redis import Redis
from app.db.session import get_db
from app.core.config import settings

router = APIRouter(tags=["health"])

@router.get("/health")
def health():
    return {"status": "ok", "service": "continuum-api"}

@router.get("/ready")
def ready(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        Redis.from_url(settings.redis_url).ping()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Dependency unavailable") from exc
    return {"status": "ready", "database": "ok", "redis": "ok"}
