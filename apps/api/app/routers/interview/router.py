"""Shared FastAPI router for interview routes."""

from fastapi import APIRouter

router = APIRouter(prefix="/interview", tags=["interview"])
