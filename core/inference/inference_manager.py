"""
AI inference module using YOLO for object detection.
Supports multiple detection types for maritime safety monitoring.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import numpy as np

from ..models import BoundingBox, Detection
from ..utils import get_logger


class Detector(ABC):
    """Abstract base class for object detectors."""

    @abstractmethod
    def load_model(self) -> bool:
        """Load detection model."""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Perform detection on a frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detections
        """
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass


class YOLODetector(Detector):
    """
    YOLO-based detector using Ultralytics.
    Supports custom maritime safety detection models.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        max_det: int = 300,
        imgsz: int = 640,
        half: bool = True
    ):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model file (.pt)
            device: Device for inference (cuda, cpu, or cuda:0)
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            max_det: Maximum number of detections
            imgsz: Input image size
            half: Use FP16 half-precision inference
        """
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.imgsz = imgsz
        self.half = half
        self.logger = get_logger()

        self._model: Optional[Any] = None
        self._loaded = False

        # Label to alarm type mapping
        self._alarm_type_map = {
            "smoking": "smoking",
            "phone": "phone_use",
            "no_helmet": "no_helmet",
            "no_lifejacket": "no_lifejacket",
            "no_workwear": "no_workwear",
            "fire": "fire",
            "smoke": "smoke",
            "fatigue": "fatigue",
            "absence": "absence"
        }

    def load_model(self) -> bool:
        """
        Load YOLO model.

        Returns:
            True if loaded successfully, False otherwise
        """
        if self._loaded:
            self.logger.warning("Model already loaded")
            return True

        try:
            from ultralytics import YOLO
            import time
            self.logger.info(f"======Loading YOLO model from {self.model_path}===========")
            t0 = time.perf_counter()
            self._model = YOLO(self.model_path)
            t1 = time.perf_counter()
            self.logger.info(f"Model loaded in {t1 - t0:.3f}s")
            # .to() and .half() are only valid for PyTorch .pt models;
            # exported formats (ONNX, TensorRT, etc.) pass device via predict()
            if self.model_path.endswith(".pt"):
                self._model.to(self.device)
                if self.half and self.device.startswith("cuda"):
                    self._model.half()

            self._loaded = True
            self.logger.info(f"YOLO model loaded successfully on {self.device}")
            return True

        except ImportError:
            self.logger.error("Ultralytics package not installed. Install with: pip install ultralytics")
            return False
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            return False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Perform detection on a frame.

        Args:
            frame: Input frame (BGR format, numpy array)

        Returns:
            List of detections (without track_id, to be assigned by tracker)
        """
        if not self._loaded or self._model is None:
            self.logger.warning("Model not loaded, cannot perform detection")
            return []

        import time  # 确保已导入 time 模块
        start_time = time.perf_counter()
        try:
            # Run inference
            results = self._model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                imgsz=self.imgsz,
                verbose=False,
                device=self.device
            )
            elapsed = time.perf_counter() - start_time


            # self.logger.info(f"========Detection elapsed: {elapsed:.3f}s============")


            if not results or len(results) == 0:
                return []

            result = results[0]
            detections = []

            # Parse results
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xywhn.cpu().numpy()  # Normalized xywh
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for i in range(len(boxes)):
                    label = self._model.names[class_ids[i]]
                    confidence = float(confidences[i])
                    bbox = BoundingBox(
                        x=float(boxes[i][0]),
                        y=float(boxes[i][1]),
                        w=float(boxes[i][2]),
                        h=float(boxes[i][3])
                    )

                    # Map label to alarm type
                    alarm_type = self._alarm_type_map.get(label)

                    detection = Detection(
                        track_id=-1,  # Will be assigned by tracker
                        label=label,
                        confidence=confidence,
                        bbox=bbox,
                        alarm_type=alarm_type
                    )
                    detections.append(detection)

            return detections

        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []

    def is_loaded(self) -> bool:
        """
        Check if model is loaded.

        Returns:
            True if loaded, False otherwise
        """
        return self._loaded


class InferenceManager:
    """
    Manages AI inference pipeline.
    Provides unified interface for detection across multiple frames.
    """

    def __init__(self, detector: Detector):
        """
        Initialize inference manager.

        Args:
            detector: Detector instance
        """
        self.detector = detector
        self.logger = get_logger()

    def initialize(self) -> bool:
        """
        Initialize inference manager (load model).

        Returns:
            True if initialized successfully, False otherwise
        """
        return self.detector.load_model()

    def infer(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detections
        """
        if not self.detector.is_loaded():
            self.logger.error("Detector not loaded")
            return []

        return self.detector.detect(frame)

    def is_ready(self) -> bool:
        """
        Check if inference manager is ready.

        Returns:
            True if ready, False otherwise
        """
        return self.detector.is_loaded()
