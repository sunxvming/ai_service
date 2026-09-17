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
        self.rtsp_transport = rtsp_transport.lower()
        self.logger = get_logger()

        self._handle = None
        self._bmcv = None
        self._decoder = None
        self._running = False
        self._thread: Optional[Thread] = None
        self._reconnect_count = 0
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_time: float = 0.0

        # --- Performance: pre-allocated reusable objects ---
        self._bmimg = None      # will be allocated once after first successful read
        self._bgr_img = None
        self._tensor = None
        self._frame_width = 0
        self._frame_height = 0

        self.setup_decoder_env()

    def setup_decoder_env(self) -> None:
        import sophon.sail as sail

        # 1. 网络传输与连接
        sail.set_decoder_env("rtsp_transport", "tcp")      # 使用TCP，避免UDP丢包导致花屏或断流
        sail.set_decoder_env("stimeout", "10000000")       # 10秒超时，避免网络波动时过早断开

        # 2. 解码缓存与性能 (核心)
        sail.set_decoder_env("extra_frame_buffer_num", "3") # 关键：大幅降低帧缓存，避免VPU内存溢出
        sail.set_decoder_env("refcounted_frames", "0")      # 由Decoder自动管理图像释放，简化代码

        # 3. 低延迟优化
        sail.set_decoder_env("low_delay", "1")             # 启用低延迟解码模式
        sail.set_decoder_env("nobuffer", "1")              # 启用无缓冲模式，进一步降低延迟

        # 4. 流解析优化 (降低首帧延迟)
        sail.set_decoder_env("probesize", "102400")        # 减小探测大小，加快拉流速度
        sail.set_decoder_env("analyzeduration", "1000000") # 减小分析时长，加快启动

        # 5. 监控场景可选优化
        sail.set_decoder_env("skip_non_idr", "2")          # 跳过非参考帧，降低解码负载 (根据监控需求决定)

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
            t0 = time.perf_counter()
            self._handle = sail.Handle(self.dev_id)
            self._bmcv = sail.Bmcv(self._handle)

            self._decoder = sail.Decoder(self.rtsp_url, True, self.dev_id)
            if not self._decoder.is_opened():
                self.logger.error(f"Sophon decoder failed to open: {self.rtsp_url}")
                self._decoder = None
                return False

            self._reconnect_count = 0
            self.logger.info(f"Sophon decoder connected to: {self.rtsp_url}")
            t1 = time.perf_counter()
            self.logger.info(f"\tSophon decoder initialized in {1000*(t1-t0):.0f}ms for {self.rtsp_url}")
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
            self._bmcv = None
            # Free reusable objects
            self._bmimg = None
            self._bgr_img = None
            self._tensor = None

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

                if ret != 0:
                    self.logger.warning(f"Sophon decoder read failed (ret={ret}) for {self.rtsp_url}")
                    if not self._reconnect():
                        break
                    continue

                # Convert with reuse of bgr_img and tensor
                frame = self._bmimage_to_numpy_reuse(self._bmimg)
                t2 = time.perf_counter()

                if frame is not None:
                    self._last_frame = frame
                    self._last_frame_time = time.perf_counter()
                    self._reconnect_count = 0
                t3 = time.perf_counter()
                # print(f"Sophon decode loop: read={1000*(t1-t0):.0f}ms, convert={1000*(t2-t1):.0f}ms, total={1000*(t3-t0):.0f}ms")
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
            print(f"asmat error: {e}")
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