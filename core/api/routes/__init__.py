"""
API routes module.
"""

from .cameras import router as cameras_router
from .health import router as health_router
from .algorithms import router as algorithms_router
from .alarm_events import router as alarm_events_router
from .scenes import router as scenes_router

__all__ = ["cameras_router", "health_router", "algorithms_router", "alarm_events_router", "scenes_router"]
