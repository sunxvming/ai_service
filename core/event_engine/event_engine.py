"""
Event engine module for alarm triggering logic.
Uses a sliding window approach: trigger when detection hits within the window
reach the alarm_threshold.
"""

import time
from collections import deque
from typing import Dict, List, Optional

from ..models import AlgorithmConfig, Detection, TrackedObject
from ..utils import get_logger


class AlarmRule:
    """Alarm rule derived from an AlgorithmConfig."""

    def __init__(self, alarm_type: str, label: str, alarm_window: int, alarm_threshold: int):
        self.alarm_type = alarm_type
        self.label = label
        self.alarm_window = max(1, alarm_window)
        self.alarm_threshold = max(1, min(alarm_threshold, self.alarm_window))


class EventEngine:
    """
    Event engine that applies a per-label sliding window to suppress false positives.
    For each (camera_id, track_id, label) tuple it maintains a boolean deque of length
    alarm_window. An alarm fires when the number of True entries >= alarm_threshold.
    After the alarm fires the window is NOT reset — subsequent frames can re-trigger
    as long as they stay above the threshold. AlarmManager is responsible for cooldown.
    """

    def __init__(self, algorithm_configs: List[AlgorithmConfig], frame_rate: int = 25):
        self.frame_rate = frame_rate
        self.logger = get_logger()

        # Build rules keyed by label (alarm_type string)
        self.rules: Dict[str, AlarmRule] = {}
        for cfg in algorithm_configs:
            # Each label in label_map that maps to this alarm type gets the same rule
            for raw_label, mapped_type in cfg.label_map.items():
                if mapped_type and mapped_type not in self.rules:
                    self.rules[mapped_type] = AlarmRule(
                        alarm_type=mapped_type,
                        label=mapped_type,
                        alarm_window=cfg.alarm_window,
                        alarm_threshold=cfg.alarm_threshold,
                    )

        # Sliding window per (camera_id, track_id, label)
        self._windows: Dict[tuple, deque] = {}
        # Last-seen time per key (for stale cleanup)
        self._last_seen: Dict[tuple, float] = {}
        # TrackedObject metadata (bbox, confidence) per key
        self._tracked: Dict[tuple, TrackedObject] = {}

        self.logger.info(f"EventEngine initialized with {len(self.rules)} alarm rules")

    def update_rules(self, algorithm_configs: List[AlgorithmConfig]) -> None:
        """Hot-reload rules from updated algorithm configs."""
        new_rules: Dict[str, AlarmRule] = {}
        for cfg in algorithm_configs:
            for raw_label, mapped_type in cfg.label_map.items():
                if mapped_type and mapped_type not in new_rules:
                    new_rules[mapped_type] = AlarmRule(
                        alarm_type=mapped_type,
                        label=mapped_type,
                        alarm_window=cfg.alarm_window,
                        alarm_threshold=cfg.alarm_threshold,
                    )
        self.rules = new_rules
        self.logger.info(f"EventEngine rules updated: {len(self.rules)} rules")

    def process_detections(
        self,
        camera_id: str,
        detections: List[Detection],
        current_time: float,
    ) -> List[Detection]:
        """
        Process one frame's detections for a camera.
        Returns detections that should trigger an alarm this frame.
        """
        active_keys = set()
        alarm_detections = []

        for det in detections:
            if det.track_id < 0:
                continue

            # Use alarm_type as the rule key; fall back to label
            rule_key = det.alarm_type or det.label
            rule = self.rules.get(rule_key)
            if not rule:
                continue

            key = (camera_id, det.track_id, rule_key)
            active_keys.add(key)
            self._last_seen[key] = current_time

            if key not in self._windows:
                self._windows[key] = deque(maxlen=rule.alarm_window)

            win = self._windows[key]

            # Resize window if rule changed
            if win.maxlen != rule.alarm_window:
                old = list(win)
                win = deque(old, maxlen=rule.alarm_window)
                self._windows[key] = win

            win.append(True)

            # Update tracked metadata
            self._tracked[key] = TrackedObject(
                track_id=det.track_id,
                label=det.label,
                bbox=det.bbox,
                confidence=det.confidence,
                alarm_type=det.alarm_type,
                frame_count=len(win),
                last_seen_time=current_time,
            )

            if self._should_trigger(win, rule):
                alarm_detections.append(det)

        # Record False for tracked-but-not-seen objects still within stale threshold
        stale_cutoff = current_time - 2.0
        stale_keys = []
        for key, last_t in self._last_seen.items():
            if key in active_keys:
                continue
            if last_t < stale_cutoff:
                stale_keys.append(key)
            else:
                # Object alive but not detected this frame
                rule_key = key[2]
                rule = self.rules.get(rule_key)
                if rule and key in self._windows:
                    self._windows[key].append(False)

        for key in stale_keys:
            self._windows.pop(key, None)
            self._last_seen.pop(key, None)
            self._tracked.pop(key, None)

        return alarm_detections

    def _should_trigger(self, window: deque, rule: AlarmRule) -> bool:
        if len(window) < rule.alarm_window:
            return False
        return sum(window) >= rule.alarm_threshold

    def reset_camera(self, camera_id: str) -> None:
        keys = [k for k in self._windows if k[0] == camera_id]
        for k in keys:
            self._windows.pop(k, None)
            self._last_seen.pop(k, None)
            self._tracked.pop(k, None)
        self.logger.info(f"EventEngine reset for camera {camera_id}")

    def reset_all(self) -> None:
        self._windows.clear()
        self._last_seen.clear()
        self._tracked.clear()
        self.logger.info("EventEngine reset all state")

    def get_tracked_count(self) -> int:
        return len(self._windows)
