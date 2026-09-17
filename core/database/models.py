"""
SQLAlchemy ORM models for database tables.
"""

from sqlalchemy import Column, String, Integer, Float, Text, DateTime
from sqlalchemy.sql import func
from .connection import Base


class IpcInfoModel(Base):
    """IpcInfo SQLAlchemy model matching the video_system database schema."""

    __tablename__ = "IpcInfo"

    IpcID        = Column(Integer, primary_key=True, autoincrement=True)
    IpcName      = Column(String(30),  nullable=False)
    NvrName      = Column(String(30),  nullable=True)
    IpcType      = Column(String(30),  nullable=True)
    OnvifAddr    = Column(String(150), nullable=True)
    ProfileToken = Column(String(50),  nullable=True)
    VideoSource  = Column(String(50),  nullable=True)
    RtspMain     = Column(String(250), nullable=True)
    RtspSub      = Column(String(250), nullable=True)
    IpcPosition  = Column(String(50),  nullable=True)
    IpcImage     = Column(String(30),  nullable=True)
    IpcX         = Column(Integer,     nullable=True)
    IpcY         = Column(Integer,     nullable=True)
    UserName     = Column(String(20),  nullable=True)
    UserPwd      = Column(String(20),  nullable=True)
    IpcEnable    = Column(String(4),   nullable=True, default="启用")
    IpcMark      = Column(String(100), nullable=True)
    scene_id     = Column(Integer,     nullable=True)
    algorithm_ids = Column(String(500), nullable=True, default=None)

    def __repr__(self):
        return f"<IpcInfo(IpcID={self.IpcID}, IpcName={self.IpcName})>"


class AlgorithmModel(Base):
    """Algorithm configuration table — one row per alarm type."""

    __tablename__ = "algorithms"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    name_cn              = Column(String(50),  nullable=False)           # 中文名，用于显示
    name_en              = Column(String(50),  nullable=False, unique=True)  # 英文名，即原config中的alarm_type
    description          = Column(String(200), nullable=True)
    model_path           = Column(String(250), nullable=False)           # PT 模型文件路径
    bmodel_path          = Column(String(250), nullable=True)            # BM1688 bmodel 路径
    bmodel_class_names   = Column(Text,        nullable=True)            # JSON: {int_idx: class_name_str}
    confidence_threshold = Column(Float,       nullable=False, default=0.5)
    label_map            = Column(Text,        nullable=False)           # JSON: {raw_label: alarm_type}
    risk_level           = Column(String(10),  nullable=False, default="中")  # 高/中/低
    alarm_cooldown       = Column(Integer,     nullable=False, default=30)   # 告警间隔（秒）
    alarm_window         = Column(Integer,     nullable=False, default=15)   # 告警窗口长度（帧数）
    alarm_threshold      = Column(Integer,     nullable=False, default=15)   # 告警阈值（窗口内命中次数）
    voice_text           = Column(String(200), nullable=True)            # 客户端语音播报内容
    enabled              = Column(Integer,     nullable=False, default=1)    # 1=启用, 0=禁用

    def __repr__(self):
        return f"<Algorithm(id={self.id}, name_en={self.name_en})>"


class AlarmEventModel(Base):
    """Alarm event record table."""

    __tablename__ = "alarm_events"

    id              = Column(Integer,     primary_key=True, autoincrement=True)
    camera_id       = Column(Integer,     nullable=False)                # 摄像头ID (IpcInfo.IpcID)
    algorithm_id    = Column(Integer,     nullable=False)                # 算法ID (algorithms.id)
    alarm_time      = Column(DateTime,    nullable=False, server_default=func.now())
    risk_level      = Column(String(10),  nullable=False, default="中")  # 高/中/低
    screenshot_path = Column(String(500), nullable=True)                 # 截图文件路径
    detection_info  = Column(Text,        nullable=True)                 # JSON: 识别框信息
    status          = Column(String(20),  nullable=False, default="未处理")  # 未处理/已查看/已处理

    def __repr__(self):
        return f"<AlarmEvent(id={self.id}, camera_id={self.camera_id}, algorithm_id={self.algorithm_id})>"


class AlarmSceneModel(Base):
    """场景配置表 - 预设算法组合"""

    __tablename__ = "alarm_scene"

    scene_id          = Column(Integer, primary_key=True, autoincrement=True)
    scene_name        = Column(String(50),  nullable=False, unique=True)
    scene_description = Column(String(200), nullable=True)
    algorithm_ids     = Column(String(500), nullable=False, default="")
    created_at        = Column(DateTime, server_default=func.now())
    updated_at        = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<AlarmScene(scene_id={self.scene_id}, scene_name={self.scene_name})>"
