"""
Multi-model inference manager for maritime safety detection.
Loads model configurations from the database (AlgorithmConfig list).
Groups algorithms by model_path so each physical model file is loaded once.
Supports two backends: "pytorch" (Ultralytics YOLO) and "sophon" (BM1688 SAIL SDK).
"""

import concurrent.futures
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

from ..models import AlgorithmConfig, BoundingBox, Detection
from ..utils import get_logger
from .inference_manager import Detector, YOLODetector


@dataclass
class ModelConfig:
    """Runtime config for a single loaded model (may cover multiple alarm types)."""
    name: str
    model_path: str          # PT path (pytorch backend)
    device: str
    confidence_threshold: float
    label_map: Dict[str, Optional[str]]  # raw_label -> alarm_type (None = ignore)
    algo_ids: Set[int] = field(default_factory=set)  # algorithm IDs served by this model
    enabled: bool = True
    bmodel_path: Optional[str] = None    # BModel path (sophon backend)
    sophon_dev_id: int = 0
    class_names: Optional[Dict[int, str]] = None  # sophon: {cls_idx: class_name}


class MappedYOLODetector(YOLODetector):
    """YOLODetector with a custom label->alarm_type mapping."""

    def __init__(self, label_map: Dict[str, Optional[str]], **kwargs):
        super().__init__(**kwargs)
        self._alarm_type_map = label_map


class MultiModelInferenceManager:
    """
    Runs multiple specialized YOLO models in parallel and merges detections.
    Supports pytorch (Ultralytics) and sophon (BM1688 SAIL) backends.
    """

    def __init__(
        self,
        model_configs: List[ModelConfig],
        max_workers: int = 10,
        inference_timeout: float = 10.0,
        use_sophon: bool = False,
    ):
        self._configs = model_configs
        self._detectors: Dict[str, Detector] = {}
        self._model_algo_ids: Dict[str, Set[int]] = {
            cfg.name: cfg.algo_ids for cfg in model_configs
        }
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._inference_timeout = inference_timeout
        self._use_sophon = use_sophon
        self.logger = get_logger()

    def initialize(self) -> bool:
        """Load all enabled models. Returns True if at least one loaded."""
        loaded = 0
        for cfg in self._configs:
            if not cfg.enabled:
                continue

            if self._use_sophon:
                detector = self._build_sophon_detector(cfg)
            else:
                detector = self._build_pytorch_detector(cfg)

            if detector is None:
                continue

            if detector.load_model():
                self._detectors[cfg.name] = detector
                self.logger.info(f"Loaded model '{cfg.name}'")
                loaded += 1
            else:
                self.logger.error(f"Failed to load model '{cfg.name}'")

        self.logger.info(
            f"MultiModelInferenceManager: {loaded}/{len(self._configs)} models loaded"
        )
        return loaded > 0

    def _build_pytorch_detector(self, cfg: "ModelConfig") -> Optional["MappedYOLODetector"]:
        if not os.path.exists(cfg.model_path):
            self.logger.warning(f"PT model '{cfg.name}' not found at {cfg.model_path}, skipping.")
            return None
        return MappedYOLODetector(
            label_map=cfg.label_map,
            model_path=cfg.model_path,
            device=cfg.device,
            confidence_threshold=cfg.confidence_threshold,
        )

    def _build_sophon_detector(self, cfg: "ModelConfig") -> Optional["SophonYOLODetector"]:
        from .sophon_detector import SophonYOLODetector
        if not cfg.bmodel_path:
            self.logger.warning(f"No bmodel_path for '{cfg.name}', skipping.")
            return None
        if not os.path.exists(cfg.bmodel_path):
            self.logger.warning(f"BModel '{cfg.name}' not found at {cfg.bmodel_path}, skipping.")
            return None
        return SophonYOLODetector(
            bmodel_path=cfg.bmodel_path,
            label_map=cfg.label_map,
            class_names=cfg.class_names,
            dev_id=cfg.sophon_dev_id,
            confidence_threshold=cfg.confidence_threshold,
            iou_threshold=0.45,
            max_det=300,
        )

    def infer_for_algorithms(self, frame: np.ndarray, algorithm_ids: Set[int]) -> List[Detection]:
        """Run only the models that serve at least one of the requested algorithm IDs."""
        if not self._detectors or not algorithm_ids:
            return []

        needed = {
            name for name, ids in self._model_algo_ids.items()
            if ids & algorithm_ids
        }
        if not needed:
            return []

        futures = {
            self._executor.submit(det.detect, frame): name
            for name, det in self._detectors.items()
            if name in needed
        }

        all_detections: List[Detection] = []
        for future, model_name in futures.items():
            try:
                all_detections.extend(future.result(timeout=self._inference_timeout))
            except concurrent.futures.TimeoutError:
                self.logger.warning(
                    f"Model '{model_name}' timed out after {self._inference_timeout}s. "
                    f"Consider increasing inference_timeout in config or using GPU (cuda)."
                )
            except Exception as e:
                self.logger.error(f"Model '{model_name}' inference error: {e}")

        return all_detections

    def is_ready(self) -> bool:
        return len(self._detectors) > 0

    def shutdown(self):
        for name, det in self._detectors.items():
            if hasattr(det, "shutdown"):
                try:
                    det.shutdown()
                except Exception as e:
                    self.logger.warning(f"Error shutting down detector '{name}': {e}")
        self._executor.shutdown(wait=False)


def build_multi_model_manager(
    algorithm_configs: List[AlgorithmConfig],
    device: str = "cpu",
    max_workers: int = 10,
    inference_timeout: float = 10.0,
    use_sophon: bool = False,
    sophon_dev_id: int = 0,
) -> MultiModelInferenceManager:
    """
    Build a MultiModelInferenceManager from a list of AlgorithmConfig objects.

    In pytorch mode (use_sophon=False), algorithms are grouped by model_path so
    each PT file is loaded only once.  In sophon mode (use_sophon=True), they are
    grouped by bmodel_path instead.

    The merged label_map is the union of all contributing algorithms' label_maps.
    The lowest confidence_threshold among the group is used (per-algo post-filter
    happens at the event-engine layer via alarm_threshold).
    """
    # Choose grouping key based on backend
    def group_key(cfg: AlgorithmConfig) -> Optional[str]:
        if use_sophon:
            return cfg.bmodel_path or None
        return cfg.model_path or None

    groups: Dict[str, List[AlgorithmConfig]] = defaultdict(list)
    for cfg in algorithm_configs:
        if not cfg.enabled:
            continue
        key = group_key(cfg)
        if key:
            groups[key].append(cfg)

    model_configs: List[ModelConfig] = []
    for path_key, algos in groups.items():
        merged_label_map: Dict[str, Optional[str]] = {}
        min_conf = min(a.confidence_threshold for a in algos)
        for algo in algos:
            merged_label_map.update(algo.label_map)

        # Merge class_names from all algorithms sharing this model.
        # bmodel_class_names is stored as JSON string: {"0": "Fire", "1": "Smoke"}
        merged_class_names: Dict[int, str] = {}
        if use_sophon:
            for algo in algos:
                raw = getattr(algo, "bmodel_class_names", None)
                if raw:
                    try:
                        import json
                        parsed = json.loads(raw) if isinstance(raw, str) else raw
                        merged_class_names.update({int(k): v for k, v in parsed.items()})
                    except Exception:
                        pass

        name = os.path.splitext(os.path.basename(path_key))[0]
        model_configs.append(ModelConfig(
            name=name,
            model_path=algos[0].model_path,
            device=device,
            confidence_threshold=min_conf,
            label_map=merged_label_map,
            algo_ids={algo.id for algo in algos},
            enabled=True,
            bmodel_path=path_key if use_sophon else None,
            sophon_dev_id=sophon_dev_id,
            class_names=merged_class_names if merged_class_names else None,
        ))

    return MultiModelInferenceManager(
        model_configs,
        max_workers=max_workers,
        inference_timeout=inference_timeout,
        use_sophon=use_sophon,
    )
