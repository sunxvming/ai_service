"""
Alarm manager: generates alarm events, applies per-algorithm cooldown,
saves screenshots, and persists records to the database.
"""

import os
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Deque, Dict, List, Optional

import cv2
import numpy as np

from ..models import AlarmEvent, AlgorithmConfig, Detection
from ..utils import get_logger

if TYPE_CHECKING:
    from ..database import AlarmEventRepository


class AlarmManager:
    """
    Manages alarm events with per-algorithm cooldown and database persistence.
    Screenshots are saved to data/screenshots/{camera_id}/{date}/{ts}_{alarm_type}.jpg.
    """

    SCREENSHOT_BASE = Path("data/screenshots")

    def __init__(
        self,
        algorithm_configs: List[AlgorithmConfig],
        alarm_event_repo: "AlarmEventRepository",
        max_alarm_cache: int = 1000,
        default_cooldown: int = 30,
    ):
        self.max_alarm_cache = max_alarm_cache
        self.default_cooldown = default_cooldown
        self.logger = get_logger()
        self._alarm_event_repo = alarm_event_repo

        # Per-algorithm configs keyed by name_en (alarm_type)
        self._algo_by_type: Dict[str, AlgorithmConfig] = {
            cfg.name_en: cfg for cfg in algorithm_configs
        }

        # In-memory recent alarm cache for quick access / TCP broadcast
        self._alarm_cache: Deque[AlarmEvent] = deque(maxlen=max_alarm_cache)

        # Cooldown tracker: {(camera_id, track_id, alarm_type): last_alarm_time}
        self._cooldown_tracker: Dict[tuple, float] = {}

        self.logger.info(
            f"AlarmManager initialized (cache: {max_alarm_cache}, "
            f"algorithms: {len(self._algo_by_type)})"
        )

    def update_configs(self, algorithm_configs: List[AlgorithmConfig]) -> None:
        """Hot-reload algorithm configs (e.g., after DB update via API)."""
        self._algo_by_type = {cfg.name_en: cfg for cfg in algorithm_configs}

    async def create_alarm(
        self,
        camera_id: str,
        camera_name: str,
        detection: Detection,
        frame: Optional[np.ndarray] = None,
        current_time: Optional[float] = None,
    ) -> Optional[AlarmEvent]:
        """
        Create an alarm if the cooldown period has passed.
        Saves a screenshot and persists the event to the database.
        Returns an AlarmEvent for TCP broadcast, or None if suppressed.
        """
        if current_time is None:
            current_time = time.time()

        if not detection.alarm_type:
            return None

        alarm_type = detection.alarm_type
        algo = self._algo_by_type.get(alarm_type)
        cooldown = algo.alarm_cooldown if algo else self.default_cooldown

        cooldown_key = (camera_id, detection.track_id, alarm_type)
        last_t = self._cooldown_tracker.get(cooldown_key)
        if last_t is not None and current_time - last_t < cooldown:
            return None

        # Save screenshot
        screenshot_path: Optional[str] = None
        if frame is not None:
            screenshot_path = self._save_screenshot(camera_id, alarm_type, frame, current_time)

        # Build AlarmEvent for in-memory cache & TCP broadcast
        alarm_id = str(uuid.uuid4())
        alarm = AlarmEvent(
            id=alarm_id,
            camera_id=camera_id,
            camera_name=camera_name,
            alarm_type=alarm_type,
            description=self._build_description(algo, detection),
            timestamp=current_time,
            track_id=detection.track_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            screenshot_path=screenshot_path,
            algorithm_id=algo.id if algo else None,
            voice_text=algo.voice_text if algo else None,
        )
        self._alarm_cache.append(alarm)
        self._cooldown_tracker[cooldown_key] = current_time

        # Persist to database
        try:
            await self._alarm_event_repo.create({
                "camera_id": int(camera_id),
                "algorithm_id": algo.id if algo else 0,
                "alarm_time": datetime.fromtimestamp(current_time),
                "risk_level": algo.risk_level if algo else "中",
                "screenshot_path": screenshot_path,
                "detection_info": {
                    "track_id": detection.track_id,
                    "label": detection.label,
                    "confidence": round(detection.confidence, 3),
                    "bbox": detection.bbox.to_list() if detection.bbox else None,
                },
                "status": "未处理",
            })
        except Exception as e:
            self.logger.error(f"Failed to persist alarm event: {e}")

        # self.logger.info(
        #     f"Alarm: {alarm_type} | camera={camera_name} "
        #     f"track={detection.track_id} conf={detection.confidence:.2f}"
        # )
        return alarm

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_screenshot(
        self,
        camera_id: str,
        alarm_type: str,
        frame: np.ndarray,
        ts: float,
    ) -> Optional[str]:
        try:
            date_str = datetime.fromtimestamp(ts).strftime("%Y%m%d")
            dir_path = self.SCREENSHOT_BASE / camera_id / date_str
            dir_path.mkdir(parents=True, exist_ok=True)
            ts_str = datetime.fromtimestamp(ts).strftime("%H%M%S_%f")[:13]
            filename = f"{ts_str}_{alarm_type}.jpg"
            full_path = dir_path / filename
            cv2.imwrite(str(full_path), frame)

            # Return path with forward slashes for HTTP URL compatibility
            relative_path = f"data/screenshots/{camera_id}/{date_str}/{filename}"
            return relative_path
        except Exception as e:
            self.logger.error(f"Failed to save screenshot: {e}")
            return None

    def _build_description(
        self, algo: Optional[AlgorithmConfig], detection: Detection
    ) -> str:
        if algo and algo.voice_text:
            base = algo.voice_text
        else:
            _defaults = {
                "smoking":       "检测到人员吸烟",
                "phone_use":     "检测到人员使用手机",
                "no_helmet":     "检测到人员未佩戴安全帽",
                "no_lifejacket": "检测到人员未穿救生衣",
                "no_workwear":   "检测到人员未穿工作服",
                "fire":          "检测到明火",
                "smoke":         "检测到烟雾",
                "fatigue":       "检测到人员疲劳",
                "absence":       "检测到岗位无人值守",
            }
            base = _defaults.get(detection.alarm_type, f"检测到异常: {detection.label}")
        return f"{base} (置信度: {detection.confidence:.2%})"

    # ------------------------------------------------------------------
    # Query helpers (used by existing code / TCP broadcast)
    # ------------------------------------------------------------------

    def get_recent_alarms(self, limit: int = 100) -> List[AlarmEvent]:
        alarms = list(self._alarm_cache)
        alarms.reverse()
        return alarms[:limit]

    def get_alarm_by_id(self, alarm_id: str) -> Optional[AlarmEvent]:
        for alarm in self._alarm_cache:
            if alarm.id == alarm_id:
                return alarm
        return None

    def get_alarm_count(self) -> int:
        return len(self._alarm_cache)

    def get_cooldown_count(self) -> int:
        return len(self._cooldown_tracker)

    def clear_cooldown(self, camera_id: Optional[str] = None) -> None:
        if camera_id is None:
            self._cooldown_tracker.clear()
        else:
            for key in [k for k in self._cooldown_tracker if k[0] == camera_id]:
                del self._cooldown_tracker[key]

    def cleanup_old_cooldowns(self, current_time: Optional[float] = None) -> None:
        if current_time is None:
            current_time = time.time()
        max_cooldown = max(
            (cfg.alarm_cooldown for cfg in self._algo_by_type.values()),
            default=self.default_cooldown,
        ) * 2
        expired = [
            k for k, t in self._cooldown_tracker.items()
            if current_time - t > max_cooldown
        ]
        for k in expired:
            del self._cooldown_tracker[k]
