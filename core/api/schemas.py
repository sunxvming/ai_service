"""
Pydantic schemas for API request/response validation.
"""

from typing import Optional, Any, Dict
from pydantic import BaseModel, Field, field_validator


# Response wrapper functions
def success_response(data: Any = None, message: str = "success") -> Dict:
    """Create a success response in the format expected by video_system client."""
    return {
        "code": 0,
        "message": message,
        "data": data
    }


def error_response(message: str, code: int = 1) -> Dict:
    """Create an error response."""
    return {
        "code": code,
        "message": message
    }


class IpcInfoBase(BaseModel):
    """Base IPC information schema."""
    ipc_name: str = Field(..., min_length=1, max_length=30)
    nvr_name: Optional[str] = Field(None, max_length=30)
    ipc_type: Optional[str] = Field(None, max_length=30)
    onvif_addr: Optional[str] = Field(None, max_length=150)
    profile_token: Optional[str] = Field(None, max_length=50)
    video_source: Optional[str] = Field(None, max_length=50)
    rtsp_main: Optional[str] = Field(None, max_length=250)
    rtsp_sub: Optional[str] = Field(None, max_length=250)
    ipc_position: Optional[str] = Field(None, max_length=50)
    ipc_image: Optional[str] = Field(None, max_length=30)
    ipc_x: Optional[int] = None
    ipc_y: Optional[int] = None
    user_name: Optional[str] = Field(None, max_length=20)
    user_pwd: Optional[str] = Field(None, max_length=20)
    ipc_enable: Optional[str] = Field("启用", max_length=4)
    ipc_mark: Optional[str] = Field(None, max_length=100)
    scene_id: Optional[int] = None
    algorithm_ids: Optional[str] = Field(None, max_length=500)

    @field_validator('rtsp_main', 'rtsp_sub')
    @classmethod
    def validate_rtsp_url(cls, v: Optional[str]) -> Optional[str]:
        if v and not v.startswith('rtsp://'):
            raise ValueError('RTSP URL must start with rtsp://')
        return v


class IpcInfoCreate(IpcInfoBase):
    """Schema for creating an IPC (IpcID is auto-generated)."""
    pass


class IpcInfoUpdate(BaseModel):
    """Schema for updating an IPC (all fields optional)."""
    ipc_name: Optional[str] = Field(None, min_length=1, max_length=30)
    nvr_name: Optional[str] = Field(None, max_length=30)
    ipc_type: Optional[str] = Field(None, max_length=30)
    onvif_addr: Optional[str] = Field(None, max_length=150)
    profile_token: Optional[str] = Field(None, max_length=50)
    video_source: Optional[str] = Field(None, max_length=50)
    rtsp_main: Optional[str] = Field(None, max_length=250)
    rtsp_sub: Optional[str] = Field(None, max_length=250)
    ipc_position: Optional[str] = Field(None, max_length=50)
    ipc_image: Optional[str] = Field(None, max_length=30)
    ipc_x: Optional[int] = None
    ipc_y: Optional[int] = None
    user_name: Optional[str] = Field(None, max_length=20)
    user_pwd: Optional[str] = Field(None, max_length=20)
    ipc_enable: Optional[str] = Field(None, max_length=4)
    ipc_mark: Optional[str] = Field(None, max_length=100)
    scene_id: Optional[int] = None
    algorithm_ids: Optional[str] = Field(None, max_length=500)

    @field_validator('rtsp_main', 'rtsp_sub')
    @classmethod
    def validate_rtsp_url(cls, v: Optional[str]) -> Optional[str]:
        if v and not v.startswith('rtsp://'):
            raise ValueError('RTSP URL must start with rtsp://')
        return v


class IpcInfoResponse(IpcInfoBase):
    """Schema for IPC response."""
    ipc_id: int

    # Runtime status (not from database)
    online: bool = False
    fps: float = 0.0

    model_config = {"from_attributes": True}

    # Mask password in response
    user_pwd: Optional[str] = Field(None, exclude=True)


class IpcPositionUpdate(BaseModel):
    """Schema for updating IPC position."""
    ipc_position: Optional[str] = Field(None, max_length=50)
    ipc_x: Optional[int] = None
    ipc_y: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    cameras_loaded: int
    cameras_online: int


# ---------------------------------------------------------------------------
# Algorithm schemas
# ---------------------------------------------------------------------------

class AlgorithmBase(BaseModel):
    name_cn: str = Field(..., min_length=1, max_length=50)
    name_en: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = Field(None, max_length=200)
    model_path: str = Field(..., min_length=1, max_length=250)
    bmodel_path: Optional[str] = Field(None, max_length=250)
    bmodel_class_names: Optional[Dict[str, Any]] = None
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    label_map: Dict[str, Any] = Field(default_factory=dict)
    risk_level: str = Field("中", pattern="^(高|中|低)$")
    alarm_cooldown: int = Field(30, ge=1, le=3600)
    alarm_window: int = Field(15, ge=1, le=1000)
    alarm_threshold: int = Field(15, ge=1, le=1000)
    voice_text: Optional[str] = Field(None, max_length=200)
    enabled: int = Field(1, ge=0, le=1)


class AlgorithmCreate(AlgorithmBase):
    pass


class AlgorithmUpdate(BaseModel):
    name_cn: Optional[str] = Field(None, min_length=1, max_length=50)
    description: Optional[str] = Field(None, max_length=200)
    model_path: Optional[str] = Field(None, min_length=1, max_length=250)
    bmodel_path: Optional[str] = Field(None, max_length=250)
    bmodel_class_names: Optional[Dict[str, Any]] = None
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    label_map: Optional[Dict[str, Any]] = None
    risk_level: Optional[str] = Field(None, pattern="^(高|中|低)$")
    alarm_cooldown: Optional[int] = Field(None, ge=1, le=3600)
    alarm_window: Optional[int] = Field(None, ge=1, le=1000)
    alarm_threshold: Optional[int] = Field(None, ge=1, le=1000)
    voice_text: Optional[str] = Field(None, max_length=200)
    enabled: Optional[int] = Field(None, ge=0, le=1)


class AlgorithmResponse(AlgorithmBase):
    id: int
    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Alarm event schemas
# ---------------------------------------------------------------------------

class AlarmEventResponse(BaseModel):
    id: int
    camera_id: int
    algorithm_id: int
    alarm_time: str
    risk_level: str
    screenshot_path: Optional[str] = None
    detection_info: Optional[Dict[str, Any]] = None
    status: str
    model_config = {"from_attributes": True}


class AlarmEventStatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(未处理|已查看|已处理)$")


class AlarmEventListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list


# ---------------------------------------------------------------------------
# AlarmScene schemas
# ---------------------------------------------------------------------------

class SceneBase(BaseModel):
    scene_name: str = Field(..., min_length=1, max_length=50)
    scene_description: Optional[str] = Field(None, max_length=200)
    algorithm_ids: str = Field("", max_length=500)


class SceneCreate(SceneBase):
    pass


class SceneUpdate(BaseModel):
    scene_name: Optional[str] = Field(None, min_length=1, max_length=50)
    scene_description: Optional[str] = Field(None, max_length=200)
    algorithm_ids: Optional[str] = Field(None, max_length=500)


class SceneResponse(SceneBase):
    scene_id: int
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    model_config = {"from_attributes": True}
