"""
FastAPI API module.
"""

from .app import create_app
from .dependencies import set_camera_manager, get_camera_manager
from .schemas import (
    IpcInfoCreate,
    IpcInfoUpdate,
    IpcInfoResponse,
    HealthResponse,
)

__all__ = [
    "create_app",
    "set_camera_manager",
    "get_camera_manager",
    "IpcInfoCreate",
    "IpcInfoUpdate",
    "IpcInfoResponse",
    "HealthResponse",
]
