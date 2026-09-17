"""
FastAPI dependencies for dependency injection.
"""

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated
    
from fastapi import Header, HTTPException, status
import os

# Global camera manager instance (set by main.py)
_camera_manager = None


def set_camera_manager(manager):
    """Set global camera manager instance."""
    global _camera_manager
    _camera_manager = manager


def get_camera_manager():
    """Get camera manager instance."""
    if _camera_manager is None:
        raise RuntimeError("Camera manager not initialized")
    return _camera_manager
