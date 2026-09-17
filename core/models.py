"""
Data models for AI Service.
Defines data structures for detection, tracking, alarms, and communication.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class AlarmType(str, Enum):
    """Alarm types matching requirements."""
    SMOKING = "smoking"
    PHONE_USE = "phone_use"
    NO_HELMET = "no_helmet"
    NO_LIFEJACKET = "no_lifejacket"
    NO_WORKWEAR = "no_workwear"
    FIRE = "fire"
    SMOKE = "smoke"
    FATIGUE = "fatigue"
    ABSENCE = "absence"


@dataclass
class BoundingBox:
    """
    Bounding box in normalized coordinates [0, 1].
    Format: (x_center, y_center, width, height)
    """
    x: float
    y: float
    w: float
    h: float

    def to_list(self) -> List[float]:
        """Convert to list format for JSON serialization."""
        return [self.x, self.y, self.w, self.h]

    @classmethod
    def from_list(cls, bbox: List[float]) -> 'BoundingBox':
        """Create from list format."""
        return cls(x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3])


@dataclass
class Detection:
    """Single detection result."""
    track_id: int
    label: str
    confidence: float
    bbox: BoundingBox
    alarm_type: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "track_id": self.track_id,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox.to_list(),
            "alarm_type": self.alarm_type
        }


@dataclass
class DetectionResult:
    """Detection result for a single frame."""
    camera_id: str
    timestamp: float
    detections: List[Detection] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": "detection",
            "camera_id": self.camera_id,
            "timestamp": int(self.timestamp),
            "detections": [det.to_dict() for det in self.detections]
        }


@dataclass
class AlarmEvent:
    """Alarm event."""
    id: str
    camera_id: str
    camera_name: str
    alarm_type: str
    description: str
    timestamp: float
    track_id: Optional[int] = None
    bbox: Optional[BoundingBox] = None
    confidence: Optional[float] = None
    screenshot_path: Optional[str] = None
    algorithm_id: Optional[int] = None
    voice_text: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = {
            "type": "alarm",
            "id": self.id,
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "alarm_type": self.alarm_type,
            "description": self.description,
            "timestamp": int(self.timestamp)
        }

        if self.track_id is not None:
            data["track_id"] = self.track_id
        if self.bbox is not None:
            data["bbox"] = self.bbox.to_list()
        if self.confidence is not None:
            data["confidence"] = round(self.confidence, 3)
        if self.screenshot_path is not None:
            data["screenshot_path"] = self.screenshot_path
        if self.algorithm_id is not None:
            data["algorithm_id"] = self.algorithm_id
        if self.voice_text is not None:
            data["voice_text"] = self.voice_text

        return data


@dataclass
class CameraStatus:
    """Camera status information."""
    camera_id: str
    online: bool
    fps: float = 0.0
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = {
            "type": "camera_status",
            "camera_id": self.camera_id,
            "online": self.online,
            "fps": round(self.fps, 2)
        }

        if self.error_message:
            data["error_message"] = self.error_message

        return data


@dataclass
class IpcInfo:
    """
    IPC (IP Camera) information matching the video_system database schema.
    This replaces CameraConfig for the new unified camera management system.
    """
    ipc_id: Optional[int] = None              # Auto-increment primary key
    ipc_name: str = ""
    nvr_name: Optional[str] = None
    ipc_type: Optional[str] = None
    onvif_addr: Optional[str] = None
    profile_token: Optional[str] = None
    video_source: Optional[str] = None
    rtsp_main: str = ""
    rtsp_sub: Optional[str] = None
    ipc_position: Optional[str] = None
    ipc_image: Optional[str] = None
    ipc_x: Optional[int] = None
    ipc_y: Optional[int] = None
    user_name: Optional[str] = None
    user_pwd: Optional[str] = None
    ipc_enable: str = "启用"                   # "启用" or "禁用"
    ipc_mark: Optional[str] = None
    scene_id: Optional[int] = None
    algorithm_ids: Optional[str] = None

    # Backward compatibility properties for CameraManager
    @property
    def id(self) -> str:
        """Return string ID for CameraManager compatibility."""
        return str(self.ipc_id) if self.ipc_id is not None else "0"

    @property
    def name(self) -> str:
        """Return name for CameraManager compatibility."""
        return self.ipc_name

    @property
    def rtsp_url(self) -> str:
        """Return main RTSP URL for CameraManager compatibility."""
        return self.rtsp_sub

    @property
    def enabled(self) -> bool:
        """Return enabled status for CameraManager compatibility."""
        return self.ipc_enable == "启用"

    @enabled.setter
    def enabled(self, value: bool):
        """Set enabled status."""
        self.ipc_enable = "启用" if value else "禁用"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "ipc_id": self.ipc_id,
            "ipc_name": self.ipc_name,
            "nvr_name": self.nvr_name,
            "ipc_type": self.ipc_type,
            "onvif_addr": self.onvif_addr,
            "profile_token": self.profile_token,
            "video_source": self.video_source,
            "rtsp_main": self.rtsp_main,
            "rtsp_sub": self.rtsp_sub,
            "ipc_position": self.ipc_position,
            "ipc_image": self.ipc_image,
            "ipc_x": self.ipc_x,
            "ipc_y": self.ipc_y,
            "user_name": self.user_name,
            "user_pwd": self.user_pwd,
            "ipc_enable": self.ipc_enable,
            "ipc_mark": self.ipc_mark,
            "scene_id": self.scene_id,
            "algorithm_ids": self.algorithm_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'IpcInfo':
        """Create from dictionary."""
        return cls(
            ipc_id=data.get("ipc_id"),
            ipc_name=data.get("ipc_name", ""),
            nvr_name=data.get("nvr_name"),
            ipc_type=data.get("ipc_type"),
            onvif_addr=data.get("onvif_addr"),
            profile_token=data.get("profile_token"),
            video_source=data.get("video_source"),
            rtsp_main=data.get("rtsp_main", ""),
            rtsp_sub=data.get("rtsp_sub"),
            ipc_position=data.get("ipc_position"),
            ipc_image=data.get("ipc_image"),
            ipc_x=data.get("ipc_x"),
            ipc_y=data.get("ipc_y"),
            user_name=data.get("user_name"),
            user_pwd=data.get("user_pwd"),
            ipc_enable=data.get("ipc_enable", "启用"),
            ipc_mark=data.get("ipc_mark"),
            scene_id=data.get("scene_id"),
            algorithm_ids=data.get("algorithm_ids"),
        )


# Keep CameraConfig as alias for backward compatibility
CameraConfig = IpcInfo


@dataclass
class AlgorithmConfig:
    """Algorithm configuration loaded from database."""
    id: int
    name_cn: str
    name_en: str                        # alarm_type identifier
    model_path: str
    confidence_threshold: float
    label_map: dict                     # {raw_model_label: alarm_type_str}
    risk_level: str                     # 高/中/低
    alarm_cooldown: int                 # seconds
    alarm_window: int                   # window length in frames
    alarm_threshold: int                # hits needed within window
    voice_text: Optional[str] = None
    description: Optional[str] = None
    enabled: int = 1
    bmodel_path: Optional[str] = None
    bmodel_class_names: Optional[str] = None  # JSON: {"0": "Fire", "1": "Smoke"}


@dataclass
class TrackedObject:
    """Tracked object with history."""
    track_id: int
    label: str
    bbox: BoundingBox
    confidence: float
    alarm_type: Optional[str] = None
    frame_count: int = 0
    first_seen_time: float = 0.0
    last_seen_time: float = 0.0
