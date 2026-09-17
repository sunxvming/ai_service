"""
Health check endpoint.
"""

from datetime import datetime
from fastapi import APIRouter, Depends

from ..dependencies import get_camera_manager
from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns service status and camera statistics.
    """
    camera_manager = get_camera_manager()

    # Get camera statistics
    all_cameras = await camera_manager.get_all_camera_configs()
    online_cameras = camera_manager.get_online_cameras()

    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow(),
        cameras_loaded=len(all_cameras),
        cameras_online=len(online_cameras)
    )
