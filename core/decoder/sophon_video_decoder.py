"""
Video decoder using Sophon SAIL SDK for hardware-accelerated decoding on BM1688.
Supports RTSP streams via VPU hardware decoder.
Optimized for minimal allocations and maximum throughput.
"""

import contextlib
import os
import time
from threading import Thread
from typing import Optional

import numpy as np

from ..utils import get_logger


@contextlib.contextmanager
def _suppress_c_stdout():
    """Redirect fd 1 (stdout) to /dev/null to silence C-level library prints."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    os.dup2(devnull, 1)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)


class SophonVideoDecoder:
    def __init__(
        self,
        rtsp_url: str,
        dev_id: int = 0,
        buffer_size: int = 10,
        reconnect_interval: int = 5,
        max_reconnect_attempts: int = -1,
        rtsp_transport: str = "tcp",
    ):
        self.rtsp_url = rtsp_url
        self.dev_id = dev_id
        self.buffer_size = buffer_size
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.logger = get_logger()

        self._handle = None
        self._decoder = None
        self._running = False
        self._thread: Optional[Thread] = None
        self._reconnect_count = 0
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_time: float = 0.0

        # --- Performance: pre-allocated reusable objects ---
        self._bmimg = None      # will be allocated once after first successful read


    def setup_decoder_env(self) -> None:
        import sophon.sail as sail

        sail.set_decoder_env("extra_frame_buffer_num", "3") # 关键：大幅降低帧缓存，避免VPU内存溢出
        sail.set_decoder_env("refcounted_frames", "0")      # 由Decoder自动管理图像释放，简化代码

        sail.set_decoder_env("low_delay", "1")             # 启用低延迟解码模式
 
    def start(self) -> bool:
        if self._running:
            self.logger.warning(f"Sophon decoder already running for {self.rtsp_url}")
            return False

        if not self._connect():
            return False

        self._running = True
        self._thread = Thread(target=self._decode_loop, daemon=True)
        self._thread.start()
        self.logger.info(f"Sophon video decoder started for {self.rtsp_url} (dev_id={self.dev_id})")
        return True

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

        self._disconnect()
        self.logger.info(f"Sophon video decoder stopped for {self.rtsp_url}")

    def _connect(self) -> bool:
        try:
            import sophon.sail as sail
            self._handle = sail.Handle(self.dev_id)
            self._decoder = sail.Decoder(self.rtsp_url, True, self.dev_id)
            if not self._decoder.is_opened():
                self.logger.error(f"Sophon decoder failed to open: {self.rtsp_url}")
                self._decoder = None
                return False

            self._reconnect_count = 0
            self.logger.info(f"Sophon decoder connected to: {self.rtsp_url}")
            return True

        except Exception as e:
            self.logger.error(f"Sophon decoder connect error for {self.rtsp_url}: {e}")
            return False

    def _disconnect(self) -> None:
        try:
            if self._decoder is not None:
                self._decoder.release()
        except Exception:
            pass
        finally:
            self._decoder = None
            self._handle = None
            self._bmimg = None


    def _decode_loop(self) -> None:
        import sophon.sail as sail

        while self._running:
            try:
                if self._decoder is None or not self._decoder.is_opened():
                    if not self._reconnect():
                        break
                    continue
                
                # Reuse BMImage if already allocated and resolution unchanged
                if self._bmimg is None:
                    self._bmimg = sail.BMImage()

                t0 = time.perf_counter()
                ret = self._decoder.read(self._handle, self._bmimg)
                t1 = time.perf_counter()
                read_cost_ms = (t1 - t0) * 1000

                if ret != 0:
                    self.logger.warning(f"Sophon decoder read failed (ret={ret}) for {self.rtsp_url}")
                    if not self._reconnect():
                        break
                    continue

                t2 = time.perf_counter()
                self._last_frame = self._bmimg.asmat()
                t3 = time.perf_counter()
                asmat_cost_ms = (t3 - t2) * 1000

                self.logger.info(
                    f"解码耗时统计 - read: {read_cost_ms:.2f} ms, asmat: {asmat_cost_ms:.2f} ms"
                )
                self._last_frame_time = time.perf_counter()
                self._reconnect_count = 0
            except Exception as e:
                self.logger.error(f"Sophon decode loop error for {self.rtsp_url}: {e}")
                if not self._reconnect():
                    break

    def _bmimage_to_numpy_reuse(self, bmimg) -> Optional[np.ndarray]:
        try:
            arr = bmimg.asmat()  # 直接获得 BGR uint8 数组
            # 如果 arr 是多维且带 batch 维度？asmat 文档明确返回 (H,W,3)，无 batch
            return arr
        except Exception as e:
            self.logger.error(f"asmat error: {e}")
            return None

    def _reconnect(self) -> bool:
        if self.max_reconnect_attempts >= 0 and self._reconnect_count >= self.max_reconnect_attempts:
            self.logger.error(
                f"Max reconnect attempts ({self.max_reconnect_attempts}) reached for {self.rtsp_url}"
            )
            return False

        self._disconnect()
        self._reconnect_count += 1
        self.logger.info(f"Sophon decoder reconnecting to {self.rtsp_url} (attempt {self._reconnect_count})...")

        time.sleep(self.reconnect_interval)

        if not self._running:
            return False

        return self._connect()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        return self._last_frame

    def is_running(self) -> bool:
        return self._running

    def get_last_frame_time(self) -> float:
        return self._last_frame_time