"""
Sophon SAIL-based YOLO detector for BM1688 hardware inference.
Uses sail.EngineImagePreProcess for hardware-accelerated preprocessing
(letterbox resize + BGR→RGB normalize) with internal thread pool for
higher Python-side throughput.

Supports two output formats:
  - NMS-free (YOLOv10/YOLOv9 end-to-end): shape (batch, max_det, 6)
    where 6 = [x1, y1, x2, y2, confidence, class_id] in model-input coords.
    No NMS required; detections are already post-processed.
  - Standard anchor (YOLOv8 and derivatives): shape (batch, 4+nc, anchors)
    or transposed (batch, anchors, 4+nc). Requires NMS.
"""

import threading
import time
from typing import Dict, List, Optional

import numpy as np

from ..models import BoundingBox, Detection
from ..utils import get_logger
from .inference_manager import Detector

# NMS-free output heuristic: if output is (batch, K, N) with K <= this value,
# treat as NMS-free (YOLOv10 style). Typical NMS-free max_det=300; typical
# standard anchor count for 640x640 is 8400.
_NMS_FREE_MAX_DET_THRESHOLD = 1000

# Channel ID used for all single-frame detect() calls.
# EngineImagePreProcess uses channel to group frames; we assign a per-instance
# unique base so multiple detectors don't clash.
_CHANNEL_COUNTER = 0
_CHANNEL_COUNTER_LOCK = threading.Lock()


def _next_channel_id() -> int:
    global _CHANNEL_COUNTER
    with _CHANNEL_COUNTER_LOCK:
        ch = _CHANNEL_COUNTER
        _CHANNEL_COUNTER += 1
    return ch


class SophonYOLODetector(Detector):
    """
    BModel-based YOLO detector using Sophon SAIL SDK.
    Uses sail.EngineImagePreProcess for on-chip letterbox resize + normalize.
    Accepts BGR numpy frames and returns Detection objects.
    Auto-detects NMS-free vs anchor-based output format.

    Thread safety: detect() is guarded by a mutex so concurrent callers
    queue through one at a time (EngineImagePreProcess is not reentrant for
    the same channel).
    """

    def __init__(
        self,
        bmodel_path: str,
        label_map: Dict[str, Optional[str]],
        class_names: Optional[Dict[int, str]] = None,
        dev_id: int = 0,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        max_det: int = 300,
    ):
        """
        Args:
            bmodel_path: Path to the compiled BModel file.
            label_map: Maps raw class name strings → alarm_type strings.
            class_names: Maps class index → raw class name string.
            dev_id: BM1688 device index (usually 0).
            confidence_threshold: Minimum detection confidence.
            iou_threshold: IoU threshold for NMS (anchor-based models only).
            max_det: Maximum number of detections to return.
        """
        self.bmodel_path = bmodel_path
        self._alarm_type_map = label_map
        self._class_names: Dict[int, str] = class_names or {}
        self.dev_id = dev_id
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.logger = get_logger()

        self._engine: object = None     # sail.EngineImagePreProcess
        self._handle: object = None     # sail.Handle
        self._bmcv: object = None       # sail.Bmcv
        self._loaded = False
        self._lock = threading.Lock()

        # Assigned a unique channel id at load time
        self._channel_id: int = 0
        self._image_index: int = 0

        # Determined at load time
        self._net_h: int = 640
        self._net_w: int = 640
        self._output_names: List[str] = []
        self._nms_free: bool = False
        self._output_transposed: bool = False

    # ------------------------------------------------------------------
    # Detector interface
    # ------------------------------------------------------------------

    def load_model(self) -> bool:
        if self._loaded:
            return True
        try:
            import sophon.sail as sail

            self._channel_id = _next_channel_id()

            # ********** 修改点 1：use_mat_output=True **********
            self._engine = sail.EngineImagePreProcess(self.bmodel_path, self.dev_id, 1)
            # BM_PADDING_TPU_LINEAR: letterbox-style padding (same as training)
            # keep_aspect_ratio=True, pre_queue=10, post_queue=10
            self._engine.InitImagePreProcess(
                sail.sail_resize_type.BM_PADDING_TPU_LINEAR,
                True, 10, 10
            )
            # Padding color: 114 grey (standard for YOLO letterbox)
            self._engine.SetPaddingAtrr(114, 114, 114, 1)

            # alpha_beta per channel: scale=1/255, shift=0 → normalize to [0,1]
            alpha_beta = (1.0 / 255, 0), (1.0 / 255, 0), (1.0 / 255, 0)
            self._engine.SetConvertAtrr(alpha_beta)

            self._net_w = self._engine.get_input_width()
            self._net_h = self._engine.get_input_height()
            self._output_names = self._engine.get_output_names()

            # Detect output format from first output tensor shape
            out_shape = list(self._engine.get_output_shape(self._output_names[0]))
            if len(out_shape) == 3 and out_shape[2] <= 10 and out_shape[1] <= _NMS_FREE_MAX_DET_THRESHOLD:
                self._nms_free = True
                self.logger.info(
                    f"SophonYOLODetector [{self.bmodel_path}]: "
                    f"NMS-free output (shape {out_shape})."
                )
            else:
                self._nms_free = False
                self._output_transposed = (len(out_shape) == 3 and out_shape[1] > out_shape[2])
                self.logger.info(
                    f"SophonYOLODetector [{self.bmodel_path}]: "
                    f"Anchor-based output (shape {out_shape}, "
                    f"transposed={self._output_transposed})."
                )

            # Handle + Bmcv for mat→BMImage conversion
            self._handle = sail.Handle(self.dev_id)
            self._bmcv = sail.Bmcv(self._handle)
            if not self._class_names:
                self.logger.warning(
                    f"SophonYOLODetector [{self.bmodel_path}]: "
                    "class_names not configured — class IDs used as raw labels."
                )

            self._loaded = True
            self.logger.info(
                f"SophonYOLODetector: loaded '{self.bmodel_path}' "
                f"(input {self._net_w}x{self._net_h}, dev_id={self.dev_id}, "
                f"conf_thresh={self.confidence_threshold}, "
                f"channel_id={self._channel_id})"
            ) 

            return True

        except Exception as e:
            self.logger.error(
                f"SophonYOLODetector: failed to load '{self.bmodel_path}': {e}"
            )
            return False

    def is_loaded(self) -> bool:
        return self._loaded

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self._loaded or self._engine is None:
            return []

        with self._lock:
            return self._detect_locked(frame)

    def _detect_locked(self, frame: np.ndarray) -> List[Detection]:
        import sophon.sail as sail

        t_start = time.perf_counter()
        try:
            ori_h, ori_w = frame.shape[:2]

            # ---------- 1. numpy BGR → sail.BMImage ----------
            t0 = time.perf_counter()
            bmimg = sail.BMImage()
            ret = self._bmcv.mat_to_bm_image(frame, bmimg)
            if ret != 0:
                raise RuntimeError(f"mat_to_bm_image failed: ret={ret}")

            # ---------- 2. Push to EngineImagePreProcess ----------
            self._image_index += 1
            img_idx = self._image_index
            while True:
                ret = self._engine.PushImage(self._channel_id, img_idx, bmimg)
                if ret == 0:
                    break
                self.logger.debug("PushImage queue full, waiting 5ms …")
                time.sleep(0.005)

            t1 = time.perf_counter()

            # ---------- 3. 修改点 2：使用 GetBatchData_Npy2 ----------
            output_tensor_map, ost_images, channel_list, imageidx_list, padding_atrr = \
                self._engine.GetBatchData_Npy2()

            t2 = time.perf_counter()

            # padding_atrr is a list of tuples, one per image in the batch.
            pad = padding_atrr[0]
            self.logger.debug(f"[detect] padding_atrr[0]={pad}")

            # 利用 padding_atrr 还原真实缩放和填充（更准确）
            # pad = [start_x, start_y, w_temp, h_temp]
            scale_x = pad[2] / ori_w
            scale_y = pad[3] / ori_h
            pad_x = pad[0]
            pad_y = pad[1]

            # ---------- 4. 修改点 3：直接按名称提取 numpy 数组 ----------
            raw_outputs = [output_tensor_map[name] for name in self._output_names]

            # ---------- 5. Post-process ----------
            t3 = time.perf_counter()
            if self._nms_free:
                dets = self._postprocess_nms_free(
                    raw_outputs, ori_w, ori_h,
                    scale_x, scale_y, pad_x, pad_y
                )
                # self.logger.info(f"[detect] NMS-free: {dets}")
            else:
                dets = self._postprocess_anchor(
                    raw_outputs, ori_w, ori_h,
                    scale_x, scale_y, pad_x, pad_y
                )
                # self.logger.info(f"[detect] Anchor-based: {dets}")
            # ---------- 6. Convert to Detection objects ----------
            t4 = time.perf_counter()
            result = self._to_detections(dets, ori_w, ori_h)
            t5 = time.perf_counter()

            model_name = self.bmodel_path.split("/")[-1]
            # self.logger.info(
            #     f"[detect] "
            #     f"push={1000*(t1-t0):.1f}ms "
            #     f"infer={1000*(t2-t1):.1f}ms "
            #     f"post={1000*(t4-t2):.1f}ms "
            #     f"total={1000*(t5-t_start):.1f}ms "
            #     f"dets={len(result)} "
            #     f"model={model_name} "
            # )
            return result

        except Exception as e:
            elapsed = (time.perf_counter() - t_start) * 1000
            self.logger.error(
                f"[detect] error after {elapsed:.1f}ms: {e}", exc_info=True
            )
            return []


    # ------------------------------------------------------------------
    # Post-processing — NMS-free (YOLOv10 / end-to-end)
    # ------------------------------------------------------------------

    def _postprocess_nms_free(
        self,
        raw_outputs: List[np.ndarray],
        ori_w: int, ori_h: int,
        scale_x: float, scale_y: float,
        pad_x: float, pad_y: float,
    ) -> np.ndarray:
        pred = raw_outputs[0][0]  # (max_det, N)
        
        # ---------- 关键修复：只取前 6 列 ----------
        if pred.shape[1] > 6:
            pred = pred[:, :6]
        
        self.logger.debug(
            f"[nms_free] conf range [{pred[:,4].min():.4f}, {pred[:,4].max():.4f}] "
            f"candidates>{self.confidence_threshold}: "
            f"{(pred[:,4] > self.confidence_threshold).sum()}"
        )
        
        mask = pred[:, 4] > self.confidence_threshold
        dets = pred[mask].copy()
        if len(dets) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        
        dets[:, [0, 2]] = (dets[:, [0, 2]] - pad_x) / scale_x
        dets[:, [1, 3]] = (dets[:, [1, 3]] - pad_y) / scale_y
        dets[:, [0, 2]] = dets[:, [0, 2]].clip(0, ori_w - 1)
        dets[:, [1, 3]] = dets[:, [1, 3]].clip(0, ori_h - 1)
        
        return dets[:self.max_det]

    # ------------------------------------------------------------------
    # Post-processing — anchor-based (YOLOv8 and derivatives)
    # ------------------------------------------------------------------

    def _postprocess_anchor(
        self,
        raw_outputs: List[np.ndarray],
        ori_w: int, ori_h: int,
        scale_x: float, scale_y: float,
        pad_x: float, pad_y: float,
    ) -> np.ndarray:
        """
        Standard anchor output: (batch, 4+nc, anchors) or transposed.
        Performs confidence filtering + NMS.
        Returns (N, 6): [x1, y1, x2, y2, score, class_id] in pixel coords.
        """
        # 如果多个输出，沿 anchor 维度拼接（假设形状为 (batch, 4+nc, anchors)）
        if len(raw_outputs) > 1:
            # 检查是否每个输出都是 (batch, 4+nc, anchors) 格式
            # 如果是 (batch, anchors, 4+nc) 则转置
            preds = []
            for out in raw_outputs:
                if out.shape[1] == out.shape[2]:  # 无法判断，但通常 (batch, 4+nc, anchors)
                    pass
                elif out.shape[1] > out.shape[2]:  # 可能是 (batch, anchors, 4+nc)，转置
                    out = out.transpose(0, 2, 1)
                preds.append(out)
            pred = np.concatenate(preds, axis=2)[0]  # 沿 anchor 维度拼接
        else:
            pred = raw_outputs[0][0]   # 单个输出，去除 batch

        # 处理 transposed 情况（单个输出时）
        if not self._output_transposed and len(raw_outputs) == 1:
            # 检查是否需要转置
            if pred.shape[0] > pred.shape[1] and pred.shape[0] > 4:
                pred = pred.T

        self.logger.debug(
            f"[anchor] pred shape after normalise={pred.shape} nc={pred.shape[0]-4}"
        )

        dets = self._nms_anchor(pred, self.confidence_threshold, self.iou_threshold)
        if dets is None or len(dets) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        dets[:, [0, 2]] = (dets[:, [0, 2]] - pad_x) / scale_x
        dets[:, [1, 3]] = (dets[:, [1, 3]] - pad_y) / scale_y
        dets[:, [0, 2]] = dets[:, [0, 2]].clip(0, ori_w - 1)
        dets[:, [1, 3]] = dets[:, [1, 3]].clip(0, ori_h - 1)

        return dets

    def _nms_anchor(
        self, pred: np.ndarray, conf_thres: float, iou_thres: float
    ) -> Optional[np.ndarray]:
        """
        pred shape: (4+nc, anchors).
        Returns (N, 6): [x1, y1, x2, y2, score, class_id].
        """
        nc = pred.shape[0] - 4
        class_scores = pred[4:4 + nc, :]
        max_scores = class_scores.max(axis=0)
        mask = max_scores > conf_thres

        self.logger.debug(
            f"[nms_anchor] anchors={pred.shape[1]} nc={nc} "
            f"candidates>{conf_thres}: {mask.sum()}"
        )

        if not mask.any():
            return None

        pred = pred[:, mask]
        boxes_cxcy = pred[:4, :].T
        scores = pred[4:4 + nc, :].T

        boxes_xyxy = self._xywh2xyxy(boxes_cxcy)
        class_ids = scores.argmax(axis=1)
        confidences = scores.max(axis=1)

        keep = self._batched_nms(boxes_xyxy, confidences, class_ids, iou_thres)
        if len(keep) == 0:
            return None

        result = np.column_stack([
            boxes_xyxy[keep],
            confidences[keep],
            class_ids[keep].astype(np.float32),
        ])
        return result[:self.max_det]

    @staticmethod
    def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
        y = x.copy()
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def _batched_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        iou_thres: float,
    ) -> np.ndarray:
        offsets = class_ids * 7680
        boxes_offset = boxes + offsets[:, None]
        return self._nms_boxes(boxes_offset, scores, iou_thres)

    @staticmethod
    def _nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[np.where(iou <= iou_thres)[0] + 1]
        return np.array(keep, dtype=np.int64)

    # ------------------------------------------------------------------
    # Map raw detections → Detection objects
    # ------------------------------------------------------------------

    def _to_detections(self, dets: np.ndarray, ori_w: int, ori_h: int) -> List[Detection]:
        detections = []
        for det in dets:
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            alarm_type = self._class_names.get(cls_id, str(cls_id))

            if alarm_type is None:
                self.logger.info(
                    f"[to_detections] cls_id={cls_id} label='{self._class_names}' "
                    f"not in label_map {list(self._alarm_type_map.keys())}"
                )

            cx = (x1 + x2) / 2.0 / ori_w
            cy = (y1 + y2) / 2.0 / ori_h
            bw = (x2 - x1) / ori_w
            bh = (y2 - y1) / ori_h

            detections.append(Detection(
                track_id=-1,
                label=alarm_type,
                confidence=float(conf),
                bbox=BoundingBox(x=float(cx), y=float(cy), w=float(bw), h=float(bh)),
                alarm_type=alarm_type,
            ))
        return detections

    def set_class_names(self, names: Dict[int, str]) -> None:
        self._class_names = names

    def shutdown(self) -> None:
        """Release EngineImagePreProcess and associated SAIL resources."""
        with self._lock:
            self._loaded = False
            self._engine = None
            self._bmcv = None
            self._handle = None
        self.logger.info(f"SophonYOLODetector: shutdown '{self.bmodel_path}'")
