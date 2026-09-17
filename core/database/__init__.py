"""
Database module for persistent storage.
"""

from .connection import init_database, create_tables, get_session, close_database
from .models import IpcInfoModel, AlgorithmModel, AlarmEventModel, AlarmSceneModel
from .camera_repository import IpcRepository, CameraRepository
from .algorithm_repository import AlgorithmRepository
from .alarm_event_repository import AlarmEventRepository
from .scene_repository import SceneRepository

__all__ = [
    "init_database",
    "create_tables",
    "get_session",
    "close_database",
    "IpcInfoModel",
    "AlgorithmModel",
    "AlarmEventModel",
    "AlarmSceneModel",
    "IpcRepository",
    "CameraRepository",
    "AlgorithmRepository",
    "AlarmEventRepository",
    "SceneRepository",
]
