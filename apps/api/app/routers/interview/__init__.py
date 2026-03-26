"""Interview Prep API: register routes on the package router."""

from app.routers.interview.router import router

# Side-effect imports attach handlers to ``router``.
from app.routers.interview import analytics, generate_evaluate, read  # noqa: F401

__all__ = ["router"]
