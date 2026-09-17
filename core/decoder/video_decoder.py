"""
Video decoder module using OpenCV for RTSP stream decoding.
Handles H.264/H.265 video streams with automatic reconnection.
"""

import time
from threading import Thread
from typing import Optional

import cv2
import numpy as np

from ..utils import get_logger


class VideoDecoder:
    """
    Video decoder for RTSP streams using OpenCV.
    Supports automatic reconnection on failure.
    """

    def __init__(
        self,
        rtsp_url: str,
        buffer_size: int = 10,
        reconnect_interval: int = 5,
        max_reconnect_attempts: int = -1,
        rtsp_transport: str = "tcp"
    ):
        """
        Initialize video decoder.

        Args:
            rtsp_url: RTSP stream URL
            buffer_size: Frame buffer size
            reconnect_interval: Reconnection interval in seconds
            max_reconnect_attempts: Max reconnection attempts (-1 for infinite)
            rtsp_transport: RTSP transport protocol (tcp or udp)
        """
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.rtsp_transport = rtsp_transport.lower()
        self.logger = get_logger()

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[Thread] = None
        self._reconnect_count = 0
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_time: float = 0.0
        self._fps: float = 0.0

    def start(self) -> bool:
        """
        Start decoding video stream.

        Returns:
            True if started successfully, False otherwise
        """
        if self._running:
            self.logger.warning(f"Decoder already running for {self.rtsp_url}")
            return False

        if not self._connect():
            return False

        self._running = True
        self._thread = Thread(target=self._decode_loop, daemon=True)
        self._thread.start()
        self.logger.info(f"Video decoder started for {self.rtsp_url}")
        return True

    def stop(self) -> None:
        """Stop decoding video stream."""
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        self._disconnect()
        self.logger.info(f"Video decoder stopped for {self.rtsp_url}")

    def _connect(self) -> bool:
        """
        Connect to RTSP stream.

        Returns:
            True if connected successfully, False otherwise
        """
        try:
            self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

            # Set RTSP transport protocol
            if self.rtsp_transport == "tcp":
                self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

            # Set buffer size
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

            if not self._cap.isOpened():
                self.logger.error(f"Failed to open RTSP stream: {self.rtsp_url}")
                return False

            # Get FPS
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._fps = fps if fps > 0 else 25.0

            self.logger.info(f"Connected to RTSP stream: {self.rtsp_url} (FPS: {self._fps})")
            self._reconnect_count = 0
            return True

        except Exception as e:
            self.logger.error(f"Error connecting to RTSP stream {self.rtsp_url}: {e}")
            return False

    def _disconnect(self) -> None:
        """Disconnect from RTSP stream."""
        if self._cap:
            self._cap.release()
            self._cap = None

    def _decode_loop(self) -> None:
        """Main decoding loop with automatic reconnection."""
        while self._running:
            try:
                if not self._cap or not self._cap.isOpened():
                    if not self._reconnect():
                        break
                    continue

                ret, frame = self._cap.read()
                if not ret or frame is None:
                    self.logger.warning(f"Failed to read frame from {self.rtsp_url}")
                    if not self._reconnect():
                        break
                    continue

                self._last_frame = frame
                self._last_frame_time = time.perf_counter()
                self._reconnect_count = 0

            except Exception as e:
                self.logger.error(f"Error in decode loop for {self.rtsp_url}: {e}")
                if not self._reconnect():
                    break

    def _reconnect(self) -> bool:
        """
        Attempt to reconnect to RTSP stream.

        Returns:
            True if reconnected or should retry, False if should stop
        """
        if self.max_reconnect_attempts >= 0 and self._reconnect_count >= self.max_reconnect_attempts:
            self.logger.error(
                f"Max reconnect attempts ({self.max_reconnect_attempts}) reached for {self.rtsp_url}"
            )
            return False

        self._disconnect()
        self._reconnect_count += 1

        self.logger.info(
            f"Reconnecting to {self.rtsp_url} (attempt {self._reconnect_count})..."
        )

        time.sleep(self.reconnect_interval)

        if not self._running:
            return False

        return self._connect()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest decoded frame.

        Returns:
            Latest frame as numpy array (BGR format), or None if no frame available
        """
        return self._last_frame

    def is_running(self) -> bool:
        """
        Check if decoder is running.

        Returns:
            True if running, False otherwise
        """
        return self._running

    def get_last_frame_time(self) -> float:
        """
        Get timestamp of last received frame.

        Returns:
            Timestamp of last frame
        """
        return self._last_frame_time
