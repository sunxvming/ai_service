"""Core package."""

from .alarm import AlarmManager
from .camera import CameraManager
from .decoder import VideoDecoder
from .event_engine import EventEngine
from .inference import InferenceManager, MultiModelInferenceManager, YOLODetector, build_multi_model_manager
from .models import (
    AlarmEvent,
    AlarmType,
    AlgorithmConfig,
    BoundingBox,
    CameraConfig,
    CameraStatus,
    Detection,
    DetectionResult,
    IpcInfo,
    TrackedObject,
)
from .process_manager import ProcessManager
from .tcp import TCPServer, InternalTCPRelay
from .tracker import ByteTracker
from .utils import Logger, get_config, get_logger, load_config

__all__ = [
    # Models
    "AlarmEvent",
    "AlarmType",
    "AlgorithmConfig",
    "BoundingBox",
    "CameraConfig",
    "CameraStatus",
    "Detection",
    "DetectionResult",
    "IpcInfo",
    "TrackedObject",
    # Utils
    "Logger",
    "get_logger",
    "load_config",
    "get_config",
    # Modules
    "TCPServer",
    "InternalTCPRelay",
    "VideoDecoder",
    "CameraManager",
    "ProcessManager",
    "YOLODetector",
    "InferenceManager",
    "MultiModelInferenceManager",
    "build_multi_model_manager",
    "ByteTracker",
    "EventEngine",
    "AlarmManager",
]
