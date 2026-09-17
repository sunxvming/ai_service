"""
Sophon SAIL-based YOLO detector for BM1688 hardware inference.
Uses sail.Engine (SYSIO mode) with numpy input/output for drop-in
compatibility with the existing pipeline.

Supports two output formats:
  - NMS-free (YOLOv10/YOLOv9 end-to-end): shape (batch, max_det, 6)
    where 6 = [x1, y1, x2, y2, confidence, class_id] in model-input coords.
    No NMS required; detections are already post-processed.
  - Standard anchor (YOLOv8 and derivatives): shape (batch, 4+nc, anchors)
    or transposed (batch, anchors, 4+nc). Requires NMS.
"""

from typing import Dict, List, Optional

import cv2
import numpy as np

from ..models import BoundingBox, Detection
from ..utils import get_logger
from .inference_manager import Detector

# NMS-free output heuristic: if output is (batch, K, N) with K <= this value,
# treat as NMS-free (YOLOv10 style). Typical NMS-free max_det=300; typical
# standard anchor count for 640x640 is 8400.
_NMS_FREE_MAX_DET_THRESHOLD = 1000


class SophonYOLODetector(Detector):
    """
    BModel-based YOLO detector using Sophon SAIL SDK.
    Accepts BGR numpy frames and returns Detection objects.
    Auto-detects NMS-free vs anchor-based output format.
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
                       E.g. {"Fire": "fire", "Smoke": "smoke"}.
            class_names: Maps class index → raw class name string.
                         E.g. {0: "Fire", 1: "Smoke"}.
                         Required for correct label mapping; if omitted,
                         class IDs are used as-is as string keys in label_map.
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
        self._handle = None          # 新增，用于Bmcv
        self._bmcv = None            # 新增
        self._net = None
        self._loaded = False

        # Determined at load time
        self._graph_name: str = ""
        self._input_name: str = ""
        self._output_names: List[str] = []
        self._net_h: int = 640
        self._net_w: int = 640
        self._input_shape: tuple = (1, 3, 640, 640)
        self._nms_free: bool = False        # True → YOLOv10 NMS-free format
        self._output_transposed: bool = False  # True → (batch, anchors, 4+nc) standard transposed

    # ------------------------------------------------------------------
    # Detector interface
    # ------------------------------------------------------------------

    def load_model(self) -> bool:
        if self._loaded:
            return True
        try:
            import sophon.sail as sail

            self._net = sail.Engine(self.bmodel_path, self.dev_id, sail.IOMode.SYSIO)
            self._graph_name = self._net.get_graph_names()[0]
            self._input_name = self._net.get_input_names(self._graph_name)[0]
            self._output_names = self._net.get_output_names(self._graph_name)
            self._input_shape = tuple(
                self._net.get_input_shape(self._graph_name, self._input_name)
            )
            self._net_h = self._input_shape[2]
            self._net_w = self._input_shape[3]

            # 新增：获取handle并初始化Bmcv
            self._handle = self._net.get_handle()
            self._bmcv = sail.Bmcv(self._handle)

            # Detect output format
            out_shape = list(self._net.get_output_shape(
                self._graph_name, self._output_names[0]
            ))
            # NMS-free: (batch, max_det≤1000, 6)  where 6 = xyxy+conf+cls
            # Standard: (batch, 4+nc, anchors)   or transposed (batch, anchors, 4+nc)
            if len(out_shape) == 3 and out_shape[1] <= _NMS_FREE_MAX_DET_THRESHOLD:
                self._nms_free = True
                self.logger.info(
                    f"SophonYOLODetector [{self.bmodel_path}]: "
                    f"NMS-free output detected (shape {out_shape}). "
                    f"Using YOLOv10-style post-processing."
                )
            else:
                self._nms_free = False
                self._output_transposed = (out_shape[1] > out_shape[2])
                self.logger.info(
                    f"SophonYOLODetector [{self.bmodel_path}]: "
                    f"Anchor-based output (shape {out_shape}, "
                    f"transposed={self._output_transposed})."
                )

            if not self._class_names:
                self.logger.warning(
                    f"SophonYOLODetector [{self.bmodel_path}]: "
                    "class_names not configured — class IDs will be used as raw labels. "
                    "Set bmodel_class_names in the algorithms table to enable correct label mapping."
                )

            self._loaded = True
            self.logger.info(
                f"SophonYOLODetector: loaded '{self.bmodel_path}' "
                f"(input {list(self._input_shape)}, dev_id={self.dev_id}, "
                f"conf_thresh={self.confidence_threshold})"
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
        import time
        start_total = time.perf_counter()
        
        if not self._loaded or self._net is None:
            elapsed = (time.perf_counter() - start_total) * 1000
            self.logger.info(f"[detect] early return (not loaded) elapsed: {elapsed:.2f}ms")
            return []
        
        try:
            ori_h, ori_w = frame.shape[:2]
            
            # ---------- 1. 预处理 ----------
            t0 = time.perf_counter()
            self.logger.info(f"[{id(self)}] detect start")

            preprocessed, ratio, (tx1, ty1) = self._preprocess(frame)
            t1 = time.perf_counter()
            preprocess_ms = (t1 - t0) * 1000
            
            # ---------- 2. 推理 (模型预测) ----------
            input_data = np.expand_dims(preprocessed, 0).astype(np.float32)
            t2 = time.perf_counter()
            raw_outputs = self._predict(input_data)
            t3 = time.perf_counter()
            predict_ms = (t3 - t2) * 1000
            
            # ---------- 3. 后处理 ----------
            t4 = time.perf_counter()
            if self._nms_free:
                dets = self._postprocess_nms_free(raw_outputs, ori_w, ori_h, ratio, tx1, ty1)
            else:
                dets = self._postprocess_anchor(raw_outputs, ori_w, ori_h, ratio, tx1, ty1)
            t5 = time.perf_counter()
            postprocess_ms = (t5 - t4) * 1000
            
            # ---------- 4. 转换为 Detection 对象 ----------
            t6 = time.perf_counter()
            result = self._to_detections(dets, ori_w, ori_h)
            t7 = time.perf_counter()
            to_detections_ms = (t7 - t6) * 1000
            
            # ---------- 总耗时 ----------
            total_ms = (t7 - start_total) * 1000
            self.logger.info(f"[{id(self)}] detect end, elapsed={time.time()-t0:.3f}s")

            # 打印各阶段耗时（INFO级别，便于查看）
            self.logger.info(
                f"[detect] timing: preprocess={preprocess_ms:.2f}ms, "
                f"predict={predict_ms:.2f}ms, postprocess={postprocess_ms:.2f}ms, "
                f"to_detections={to_detections_ms:.2f}ms, total={total_ms:.2f}ms, "
                f"detections={len(result)}"
            )
            
            if result:
                self.logger.info(
                    f"[detect] {len(result)} detections: "
                    + ", ".join(f"{d.label}({d.confidence:.2f})" for d in result)
                )
            
            return result
            
        except Exception as e:
            elapsed = (time.perf_counter() - start_total) * 1000
            self.logger.error(f"[detect] error: {e}, elapsed: {elapsed:.2f}ms", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess(self, img: np.ndarray):
        """
        使用 Bmcv 硬件加速缩放，返回 NCHW 格式的 RGB 归一化数组。
        """
        import sophon.sail as sail

        target_h = self._net_h
        target_w = self._net_w

        try:
            # 1. numpy -> BMImage (BGR packed)
            bmimg = sail.BMImage()
            ret = self._bmcv.mat_to_bm_image(img, bmimg)
            if ret != 0:
                raise RuntimeError("mat_to_bm_image failed")

            # 2. 硬件缩放 (直接拉伸至模型输入尺寸)
            resized_bmimg = sail.BMImage()
            ret = self._bmcv.vpp_resize(
                bmimg,
                resized_bmimg,
                target_w,
                target_h,
                sail.bmcv_resize_algorithm.BMCV_INTER_NEAREST
            )
            if ret != 0:
                raise RuntimeError("vpp_resize failed")

            # 3. 转为 Tensor (设备→主机)
            tensor = self._bmcv.bm_image_to_tensor(resized_bmimg)
            arr = tensor.asnumpy()  # 期望 shape: (1, 3, H, W) 即 NCHW

            # 4. 调整形状并归一化
            if arr.ndim == 4:
                chw = arr[0]                # (C, H, W)
                chw = chw[::-1].copy()      # BGR → RGB (反转通道)
                chw = chw.astype(np.float32) / 255.0
                return chw, (1.0, 1.0), (0, 0)
            else:
                raise RuntimeError(f"Unexpected tensor shape: {arr.shape}")

        except Exception as e:
            self.logger.warning(
                f"Hardware preprocess failed ({e}), falling back to CPU letterbox."
            )

    @staticmethod
    def _letterbox(im: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
        h, w = im.shape[:2]
        r = min(new_shape[0] / h, new_shape[1] / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw = (new_shape[1] - new_unpad[0]) / 2
        dh = (new_shape[0] - new_unpad[1]) / 2
        if (w, h) != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return im, (r, r), (dw, dh)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict(self, input_data: np.ndarray) -> List[np.ndarray]:
        outputs = self._net.process(self._graph_name, {self._input_name: input_data})
        # Re-order outputs to match self._output_names order
        out_keys = list(outputs.keys())
        ordered = []
        for name in self._output_names:
            for k in out_keys:
                if name == k:
                    ordered.append(outputs[k])
                    break
        return ordered

    # ------------------------------------------------------------------
    # Post-processing — NMS-free (YOLOv10 / end-to-end)
    # ------------------------------------------------------------------

    def _postprocess_nms_free(
        self,
        raw_outputs: List[np.ndarray],
        ori_w: int, ori_h: int,
        ratio: tuple, tx1: float, ty1: float,
    ) -> np.ndarray:
        """
        YOLOv10 NMS-free output: (batch, max_det, 6)
        Each row: [x1, y1, x2, y2, confidence, class_id]
        Coordinates are in model-input pixel space (0 to net_w/net_h).
        """
        pred = raw_outputs[0][0]   # (max_det, 6)

        self.logger.debug(
            f"[nms_free] pred shape={pred.shape}  "
            f"conf range: {pred[:, 4].min():.4f} ~ {pred[:, 4].max():.4f}  "
            f"candidates > {self.confidence_threshold}: "
            f"{(pred[:, 4] > self.confidence_threshold).sum()}"
        )

        # Filter by confidence
        mask = pred[:, 4] > self.confidence_threshold
        dets = pred[mask].copy()

        if len(dets) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        # Rescale xyxy from model-input space back to original image
        dets[:, [0, 2]] -= tx1
        dets[:, [1, 3]] -= ty1
        dets[:, [0, 2]] /= ratio[0]
        dets[:, [1, 3]] /= ratio[1]
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
        ratio: tuple, tx1: float, ty1: float,
    ) -> np.ndarray:
        """
        Standard anchor output: (batch, 4+nc, anchors) or transposed.
        Performs confidence filtering + NMS.
        Returns (N, 6): [x1, y1, x2, y2, score, class_id] in pixel coords.
        """
        pred = np.concatenate(raw_outputs, axis=0)[0]  # remove batch dim

        # Normalise to (4+nc, anchors)
        if self._output_transposed:
            pred = pred.T

        self.logger.debug(
            f"[anchor] pred shape after normalise={pred.shape}  "
            f"nc={pred.shape[0]-4}"
        )

        dets = self._nms_anchor(pred, self.confidence_threshold, self.iou_threshold)
        if dets is None or len(dets) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        # Rescale
        dets[:, [0, 2]] -= tx1
        dets[:, [1, 3]] -= ty1
        dets[:, [0, 2]] /= ratio[0]
        dets[:, [1, 3]] /= ratio[1]
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
        class_scores = pred[4:4 + nc, :]       # (nc, anchors)
        max_scores = class_scores.max(axis=0)   # (anchors,)
        mask = max_scores > conf_thres

        self.logger.debug(
            f"[nms_anchor] anchors={pred.shape[1]}  nc={nc}  "
            f"candidates>{conf_thres}: {mask.sum()}"
        )

        if not mask.any():
            return None

        pred = pred[:, mask]
        boxes_cxcy = pred[:4, :].T             # (N, 4) cx cy w h
        scores = pred[4:4 + nc, :].T           # (N, nc)

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
            label = self._class_names.get(cls_id, str(cls_id))
            alarm_type = self._alarm_type_map.get(label)

            if alarm_type is None:
                self.logger.debug(
                    f"[to_detections] cls_id={cls_id} label='{label}' "
                    f"has no alarm_type in label_map {list(self._alarm_type_map.keys())}"
                )

            cx = (x1 + x2) / 2.0 / ori_w
            cy = (y1 + y2) / 2.0 / ori_h
            bw = (x2 - x1) / ori_w
            bh = (y2 - y1) / ori_h

            detections.append(Detection(
                track_id=-1,
                label=label,
                confidence=float(conf),
                bbox=BoundingBox(x=float(cx), y=float(cy), w=float(bw), h=float(bh)),
                alarm_type=alarm_type,
            ))
        return detections

    def set_class_names(self, names: Dict[int, str]) -> None:
        """Set class index → name mapping (replaces any existing mapping)."""
        self._class_names = names
