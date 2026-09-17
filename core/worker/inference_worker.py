"""
Inference worker process.

Handles RTSP decoding, model inference, object tracking, event suppression,
and alarm management for a single camera. Results are forwarded to the main
process via a TCP connection to the internal relay on port 9003.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Set, Union

from ..models import AlgorithmConfig, CameraStatus, DetectionResult
from ..database import init_database, CameraRepository, AlgorithmRepository, AlarmEventRepository
from ..decoder import VideoDecoder, SophonVideoDecoder
from ..inference.multi_model_inference import build_multi_model_manager
from ..tracker import ByteTracker
from ..event_engine import EventEngine
from ..alarm import AlarmManager
from ..utils import Logger, get_logger, load_config


# Python 3.8 compat: asyncio.to_thread added in 3.9
if sys.version_info >= (3, 9):
    _async_to_thread = asyncio.to_thread
else:
    _executor = ThreadPoolExecutor()

    async def _async_to_thread(func, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, lambda: func(*args, **kwargs))


def _row_to_config(row: dict) -> AlgorithmConfig:
    return AlgorithmConfig(
        id=row["id"],
        name_cn=row["name_cn"],
        name_en=row["name_en"],
        model_path=row["model_path"],
        confidence_threshold=row["confidence_threshold"],
        label_map=row["label_map"],
        risk_level=row["risk_level"],
        alarm_cooldown=row["alarm_cooldown"],
        alarm_window=row["alarm_window"],
        alarm_threshold=row["alarm_threshold"],
        voice_text=row.get("voice_text"),
        description=row.get("description"),
        enabled=row.get("enabled", 1),
        bmodel_path=row.get("bmodel_path"),
        bmodel_class_names=row.get("bmodel_class_names"),
    )


class InferenceWorker:
    """
    Runs in a dedicated child process for one camera.

    Lifecycle:
      1. Load config and init logger (per-process, isolated)
      2. Open own DB connection; load camera and algorithm configs
      3. Start RTSP decoder
      4. Load inference models for this camera's algorithms only
      5. Connect to main-process relay on relay_port
      6. Run frame loop: infer → track → event engine → alarm → send
    """

    STATUS_INTERVAL = 5.0   # seconds between camera-status broadcasts
    RELAY_RETRY_INTERVAL = 3.0  # seconds between relay reconnection attempts

    def __init__(
        self,
        ipc_id: int,
        config_path: str,
        relay_host: str,
        relay_port: int,
    ) -> None:
        self._ipc_id = ipc_id
        self._config_path = config_path
        self._relay_host = relay_host
        self._relay_port = relay_port

        self.config = None
        self.logger = None

        self._camera = None
        self._algorithm_configs: List[AlgorithmConfig] = []
        self._algorithm_ids: Set[int] = set()
        self._allowed_alarm_types: Set[str] = set()

        self._decoder: Optional[Union[VideoDecoder, SophonVideoDecoder]] = None
        self._inference_manager = None
        self._tracker: Optional[ByteTracker] = None
        self._event_engine: Optional[EventEngine] = None
        self._alarm_manager: Optional[AlarmManager] = None

        self._relay_writer: Optional[asyncio.StreamWriter] = None
        self._running = False

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        try:
            await self._initialize()
        except Exception as e:
            print(f"[Worker cam{self._ipc_id}] Initialization failed: {e}", flush=True)
            return

        self._running = True
        self.logger.info(f"Worker cam{self._ipc_id} entering main loop")
        try:
            await asyncio.gather(
                self._relay_reconnect_loop(),
                self._processing_loop(),
                self._status_broadcast_loop(),
            )
        except (KeyboardInterrupt, SystemExit):
            pass
        except Exception as e:
            if self.logger:
                self.logger.error(f"Worker cam{self._ipc_id} fatal error: {e}")
        finally:
            self._running = False
            await self._shutdown()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def _initialize(self) -> None:
        self.config = load_config(self._config_path)

        # Give each worker its own log file so writes don't interleave
        log_config = self.config.get_logging_config()
        log_file = log_config.get("log_file", "ai_service.log")
        stem, _, suffix = log_file.rpartition(".")
        log_config["log_file"] = f"{stem}_cam{self._ipc_id}.{suffix or 'log'}"
        Logger.initialize(**log_config)
        self.logger = get_logger()
        self.logger.info(f"Worker starting for camera {self._ipc_id}")

        # Own DB connection — processes must not share SQLAlchemy engines
        db_config = self.config.get("database", {})
        db_path = db_config.get("path", "data/cameras.db")
        init_database(db_path)

        camera_repo = CameraRepository()
        algorithm_repo = AlgorithmRepository()
        alarm_event_repo = AlarmEventRepository()

        self._camera = await camera_repo.get_by_id(self._ipc_id)
        if self._camera is None:
            raise RuntimeError(f"Camera {self._ipc_id} not found in DB")
        if not self._camera.enabled:
            raise RuntimeError(f"Camera {self._ipc_id} is disabled")

        # Resolve algorithm IDs assigned to this camera
        if self._camera.algorithm_ids:
            ids_str = self._camera.algorithm_ids.strip()
            if ids_str:
                try:
                    self._algorithm_ids = {int(x.strip()) for x in ids_str.split(",") if x.strip()}
                except ValueError:
                    self.logger.warning(f"Invalid algorithm_ids for camera {self._ipc_id}")

        # Load only the algorithms this camera uses
        all_rows = await algorithm_repo.get_all(enabled_only=True)
        all_configs = [_row_to_config(r) for r in all_rows]
        self._algorithm_configs = (
            [c for c in all_configs if c.id in self._algorithm_ids]
            if self._algorithm_ids
            else []
        )
        self.logger.info(
            f"Camera {self._ipc_id}: {len(self._algorithm_configs)} algorithm(s) assigned"
        )

        # Collect the alarm types this camera is permitted to fire
        for algo in self._algorithm_configs:
            for alarm_type in algo.label_map.values():
                if alarm_type:
                    self._allowed_alarm_types.add(alarm_type)

        # RTSP decoder — select backend from config
        decoder_config = self.config.get_decoder_config()
        use_sophon = decoder_config.get("backend", "opencv").lower() == "sophon"
        if use_sophon:
            self._decoder = SophonVideoDecoder(
                rtsp_url=self._camera.rtsp_url,
                dev_id=decoder_config.get("sophon", {}).get("dev_id", 0),
                buffer_size=decoder_config.get("buffer_size", 10),
                reconnect_interval=decoder_config.get("reconnect_interval", 5),
                max_reconnect_attempts=decoder_config.get("max_reconnect_attempts", -1),
                rtsp_transport=decoder_config.get("rtsp_transport", "tcp"),
            )
        else:
            self._decoder = VideoDecoder(
                rtsp_url=self._camera.rtsp_url,
                buffer_size=decoder_config.get("buffer_size", 10),
                reconnect_interval=decoder_config.get("reconnect_interval", 5),
                max_reconnect_attempts=decoder_config.get("max_reconnect_attempts", -1),
                rtsp_transport=decoder_config.get("rtsp_transport", "tcp"),
            )
        backend_name = "Sophon VPU" if use_sophon else "OpenCV"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._decoder.start)
        self.logger.info(
            f"===Camera {self._ipc_id}: {backend_name} decoder started ({self._camera.rtsp_url})==="
        )

        # Inference — load only this camera's models
        inference_config = self.config.get("inference", {})
        self._inference_manager = build_multi_model_manager(
            algorithm_configs=self._algorithm_configs,
            device=inference_config.get("device", "cpu"),
            inference_timeout=inference_config.get("inference_timeout", 10.0),
            use_sophon=use_sophon,
        )
        if self._algorithm_configs and not self._inference_manager.initialize():
            self.logger.warning(f"Camera {self._ipc_id}: no models loaded — detections will be empty")

        tracker_config = self.config.get_tracker_config()

        self._tracker = ByteTracker(
            track_thresh=tracker_config.get("track_thresh", 0.5),
            track_buffer=tracker_config.get("track_buffer", 30),
            match_thresh=tracker_config.get("match_thresh", 0.8),
            frame_rate=tracker_config.get("frame_rate", 25),
        )

        self._event_engine = EventEngine(
            algorithm_configs=self._algorithm_configs,
            frame_rate=tracker_config.get("frame_rate", 25),
        )

        self._alarm_manager = AlarmManager(
            algorithm_configs=self._algorithm_configs,
            alarm_event_repo=alarm_event_repo,
            max_alarm_cache=1000,
            default_cooldown=30,
        )
        self.logger.info(f"============================================")
        self.logger.info(f"Worker cam{self._ipc_id} initialized")
        self.logger.info(f"============================================")

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------

    async def _relay_reconnect_loop(self) -> None:
        """Keep a TCP connection open to the main-process relay."""
        self.logger.info(f"Worker cam{self._ipc_id} starting relay reconnect loop")
        while self._running:
            if self._relay_writer is None or self._relay_writer.is_closing():
                try:
                    _, writer = await asyncio.open_connection(
                        self._relay_host, self._relay_port
                    )
                    self._relay_writer = writer
                    self.logger.info(
                        f"Worker cam{self._ipc_id}: connected to relay "
                        f"{self._relay_host}:{self._relay_port}"
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.warning(
                        f"Worker cam{self._ipc_id}: relay connect failed ({e}), "
                        f"retrying in {self.RELAY_RETRY_INTERVAL}s"
                    )
                    await asyncio.sleep(self.RELAY_RETRY_INTERVAL)
                    continue
            await asyncio.sleep(self.RELAY_RETRY_INTERVAL)

    async def _processing_loop(self) -> None:
        self.logger.info(f"Worker cam{self._ipc_id} starting processing loop")
        frame_interval = 1.0 / 25.0
        while self._running:
            try:
                t0 = time.time()
                await self._process_frame()
                # print(f"===Worker cam{self._ipc_id} processing time: {time.time() - t0:.3f}s")
                self._alarm_manager.cleanup_old_cooldowns()
                sleep_time = max(0.0, frame_interval - (time.time() - t0))
                if sleep_time:
                    await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error(f"Worker cam{self._ipc_id} processing error: {e}")
                await asyncio.sleep(1)

    async def _status_broadcast_loop(self) -> None:
        self.logger.info(f"Worker cam{self._ipc_id} starting status broadcast loop")
        while self._running:
            try:
                await asyncio.sleep(self.STATUS_INTERVAL)
                if self._decoder is None:
                    continue
                online = self._decoder.is_running()
                status = CameraStatus(
                    camera_id=str(self._ipc_id),
                    online=online,
                    fps=0.0,
                )
                await self._send_message(status.to_dict())
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error(f"Worker cam{self._ipc_id} status error: {e}")

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    async def _process_frame(self) -> None:
        if self._decoder is None:
            return
        frame = self._decoder.get_latest_frame()
        if frame is None:
            return
        if not self._algorithm_ids:
            return

        current_time = time.time()

        detections = await _async_to_thread(self._infer_with_filter, frame)
        if not detections:
            return

        tracked = self._tracker.update(detections, frame)
        if not tracked:
            return

        result = DetectionResult(
            camera_id=str(self._ipc_id),
            timestamp=current_time,
            detections=tracked,
        )
        await self._send_message(result.to_dict())
        # print DetectionResult
        # print(f"===Worker cam{self._ipc_id} detections: {len(tracked)} objects")
        # print(f"\t detections: {tracked}")

        # EventEngine is stateful and fast — call synchronously to preserve ordering
        alarm_detections = self._event_engine.process_detections(
            str(self._ipc_id), tracked, current_time
        )
        if not alarm_detections:
            return

        # print(f"===Worker cam{self._ipc_id} alarm detections: {len(alarm_detections)} objects")
        # print(f"\t alarm detections: {alarm_detections}")

        # Fire alarm creation in background so it doesn't block next frame
        asyncio.create_task(self._create_alarms(alarm_detections, frame, current_time))

    async def _create_alarms(self, alarm_detections: list, frame, current_time: float) -> None:
        camera_name = self._camera.ipc_name if self._camera else str(self._ipc_id)
        for det in alarm_detections:
            try:
                alarm = await self._alarm_manager.create_alarm(
                    camera_id=str(self._ipc_id),
                    camera_name=camera_name,
                    detection=det,
                    frame=frame,
                    current_time=current_time,
                )
                if alarm:
                    await self._send_message(alarm.to_dict())
            except Exception as e:
                self.logger.error(f"Worker cam{self._ipc_id} create_alarm error: {e}")

    def _infer_with_filter(self, frame) -> list:
        if not self._algorithm_ids or self._inference_manager is None:
            return []
        all_detections = self._inference_manager.infer_for_algorithms(frame, self._algorithm_ids)
        return [d for d in all_detections if d.alarm_type in self._allowed_alarm_types]

    # ------------------------------------------------------------------
    # Relay messaging
    # ------------------------------------------------------------------

    async def _send_message(self, msg: dict) -> None:
        writer = self._relay_writer
        if writer is None or writer.is_closing():
            return
        try:
            data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
            writer.write(data)
            await writer.drain()
        except Exception:
            self._relay_writer = None

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def _shutdown(self) -> None:
        if self._decoder:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._decoder.stop)
            except Exception:
                pass
        writer = self._relay_writer
        if writer and not writer.is_closing():
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        if self.logger:
            self.logger.info(f"Worker cam{self._ipc_id} stopped")


# ------------------------------------------------------------------
# Module-level entry point (must be importable for multiprocessing spawn)
# ------------------------------------------------------------------

def run_inference_worker(
    ipc_id: int,
    config_path: str,
    relay_host: str,
    relay_port: int,
) -> None:
    """Entry point for multiprocessing.Process(target=run_inference_worker)."""
    # Ensure working-directory-relative paths resolve correctly in the child process
    import os
    script_dir = Path(__file__).resolve().parent.parent.parent  # ai_service/
    os.chdir(script_dir)

    asyncio.run(InferenceWorker(ipc_id, config_path, relay_host, relay_port).run())
