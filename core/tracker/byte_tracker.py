"""
Multi-object tracker module using ByteTrack algorithm.
Assigns and maintains track IDs across frames.
"""

from typing import Dict, List, Optional

import numpy as np

from ..models import BoundingBox, Detection
from ..utils import get_logger


class ByteTracker:
    """
    ByteTrack implementation for multi-object tracking.
    Maintains track IDs and associations across frames.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 25
    ):
        """
        Initialize ByteTrack tracker.

        Args:
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to keep lost tracks
            match_thresh: Matching threshold for track association
            frame_rate: Frame rate for track management
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.logger = get_logger()

        self._next_track_id = 1
        self._active_tracks: Dict[int, Track] = {}
        self._lost_tracks: Dict[int, Track] = {}
        self._frame_count = 0

        # Try to import ByteTrack from various sources
        self._tracker_impl = None
        self._init_tracker()

    def _init_tracker(self) -> None:
        """Initialize ByteTrack implementation."""
        self._tracker_impl = None

        # try:
        #     # Try importing from boxmot (preferred)
        #     from boxmot import BYTETracker as BoxmotByteTrack
        #     self._tracker_impl = BoxmotByteTrack(
        #         track_thresh=self.track_thresh,
        #         track_buffer=self.track_buffer,
        #         match_thresh=self.match_thresh,
        #         frame_rate=self.frame_rate
        #     )
        #     self.logger.info("Using boxmot ByteTrack implementation")
        # except ImportError:
        #     self.logger.warning("boxmot not available, using simple tracker fallback")
        #     self._tracker_impl = None

    def update(self, detections: List[Detection], frame: Optional[np.ndarray] = None) -> List[Detection]:
        """
        Update tracker with new detections and assign track IDs.

        Args:
            detections: List of detections from current frame
            frame: Current frame (optional, for visual trackers)

        Returns:
            List of detections with assigned track IDs
        """
        self._frame_count += 1

        if not detections:
            # Update lost tracks
            self._update_lost_tracks()
            return []

        if self._tracker_impl is not None:
            # Use boxmot ByteTrack
            return self._update_with_boxmot(detections, frame)
        else:
            # Use simple IOU-based tracker
            return self._update_simple(detections)

    def _update_with_boxmot(self, detections: List[Detection], frame: Optional[np.ndarray]) -> List[Detection]:
        """
        Update using boxmot ByteTrack implementation.

        Args:
            detections: List of detections
            frame: Current frame

        Returns:
            List of detections with track IDs
        """
        try:
            if not detections:
                return []

            if frame is None:
                return self._update_simple(detections)

            h, w = frame.shape[:2]

            # Convert detections to numpy format [x1, y1, x2, y2, conf, class_id]
            # Keep pixel-coordinate xyxy alongside each detection for matching later
            dets_xyxy = []
            dets_array = []
            for det in detections:
                x1 = (det.bbox.x - det.bbox.w / 2) * w
                y1 = (det.bbox.y - det.bbox.h / 2) * h
                x2 = (det.bbox.x + det.bbox.w / 2) * w
                y2 = (det.bbox.y + det.bbox.h / 2) * h
                dets_xyxy.append((x1, y1, x2, y2))
                dets_array.append([x1, y1, x2, y2, det.confidence, 0])

            dets_np = np.array(dets_array, dtype=np.float32)

            # Update tracker — returned rows: [x1, y1, x2, y2, track_id, conf, class_id, ...]
            tracks = self._tracker_impl.update(dets_np, frame)

            if tracks is None or (hasattr(tracks, 'size') and tracks.size == 0):
                return detections

            # Match each track back to the input detection with the highest IOU.
            # boxmot may reorder / filter rows, so index-based mapping is wrong.
            for track in tracks:
                tx1, ty1, tx2, ty2 = track[0], track[1], track[2], track[3]
                track_id = int(track[4])

                best_idx = -1
                best_iou = 0.0
                for i, (dx1, dy1, dx2, dy2) in enumerate(dets_xyxy):
                    inter_x1 = max(tx1, dx1)
                    inter_y1 = max(ty1, dy1)
                    inter_x2 = min(tx2, dx2)
                    inter_y2 = min(ty2, dy2)
                    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                        continue
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    t_area = (tx2 - tx1) * (ty2 - ty1)
                    d_area = (dx2 - dx1) * (dy2 - dy1)
                    union_area = t_area + d_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i

                if best_idx >= 0:
                    detections[best_idx].track_id = track_id

            return detections

        except Exception as e:
            self.logger.error(f"Error in boxmot tracking: {e}, falling back to simple tracker")
            return self._update_simple(detections)

    def _update_simple(self, detections: List[Detection]) -> List[Detection]:
        """
        Simple IOU-based tracker fallback with distance metric for small objects.

        Args:
            detections: List of detections

        Returns:
            List of detections with track IDs
        """
        # Match detections with active tracks using hybrid IOU + distance metric
        matched_detections = []
        unmatched_detections = list(detections)

        for track_id, track in list(self._active_tracks.items()):
            best_match_idx = -1
            best_score = 0.0

            for i, det in enumerate(unmatched_detections):
                # For small objects (area < 0.01), use center distance weighted with IOU
                det_area = det.bbox.w * det.bbox.h
                track_area = track.bbox.w * track.bbox.h
                is_small = det_area < 0.01 or track_area < 0.01

                if is_small:
                    # Center distance metric (normalized)
                    cx_dist = abs(det.bbox.x - track.bbox.x)
                    cy_dist = abs(det.bbox.y - track.bbox.y)
                    center_dist = (cx_dist ** 2 + cy_dist ** 2) ** 0.5

                    # Convert distance to similarity (closer = higher score)
                    # Threshold: 0.1 normalized distance (~10% of image) is "near"
                    dist_score = max(0, 1.0 - center_dist / 0.1)

                    # IOU as secondary metric
                    iou = self._calculate_iou(track.bbox, det.bbox)

                    # Weighted combination: 70% distance, 30% IOU for small objects
                    score = 0.7 * dist_score + 0.3 * iou
                    threshold = 0.5  # More lenient for small objects
                else:
                    # Regular IOU matching for normal-sized objects
                    score = self._calculate_iou(track.bbox, det.bbox)
                    threshold = self.match_thresh

                if score > best_score and score > threshold:
                    best_score = score
                    best_match_idx = i

            if best_match_idx >= 0:
                # Match found
                matched_det = unmatched_detections.pop(best_match_idx)
                matched_det.track_id = track_id
                matched_detections.append(matched_det)

                # Update track
                track.bbox = matched_det.bbox
                track.age = 0
            else:
                # Track lost
                track.age += 1
                if track.age > self.track_buffer:
                    del self._active_tracks[track_id]

        # Assign new track IDs to unmatched detections
        for det in unmatched_detections:
            det.track_id = self._next_track_id
            self._active_tracks[self._next_track_id] = Track(
                track_id=self._next_track_id,
                bbox=det.bbox,
                age=0
            )
            self._next_track_id += 1
            matched_detections.append(det)

        return matched_detections

    def _update_lost_tracks(self) -> None:
        """Update age of lost tracks and remove expired ones."""
        for track_id in list(self._active_tracks.keys()):
            track = self._active_tracks[track_id]
            track.age += 1
            if track.age > self.track_buffer:
                del self._active_tracks[track_id]

    @staticmethod
    def _calculate_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """
        Calculate IOU between two bounding boxes.

        Args:
            bbox1: First bounding box (normalized xywh)
            bbox2: Second bounding box (normalized xywh)

        Returns:
            IOU value [0, 1]
        """
        # Convert xywh to xyxy
        x1_min = bbox1.x - bbox1.w / 2
        y1_min = bbox1.y - bbox1.h / 2
        x1_max = bbox1.x + bbox1.w / 2
        y1_max = bbox1.y + bbox1.h / 2

        x2_min = bbox2.x - bbox2.w / 2
        y2_min = bbox2.y - bbox2.h / 2
        x2_max = bbox2.x + bbox2.w / 2
        y2_max = bbox2.y + bbox2.h / 2

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Calculate union
        bbox1_area = bbox1.w * bbox1.h
        bbox2_area = bbox2.w * bbox2.h
        union_area = bbox1_area + bbox2_area - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def reset(self) -> None:
        """Reset tracker state."""
        self._next_track_id = 1
        self._active_tracks.clear()
        self._lost_tracks.clear()
        self._frame_count = 0
        self.logger.info("Tracker reset")


class Track:
    """Simple track representation for fallback tracker."""

    def __init__(self, track_id: int, bbox: BoundingBox, age: int = 0):
        self.track_id = track_id
        self.bbox = bbox
        self.age = age
