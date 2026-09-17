"""
Ship Intelligent Monitoring System - AI Service
Main process entry point.

Responsibilities:
  - HTTP API (FastAPI/uvicorn, port 9001)
  - Qt broadcast TCP server (port 9002)
  - Internal relay TCP server (port 9003) — receives from worker processes
  - ProcessManager — one inference child process per enabled camera
"""

import asyncio
import multiprocessing
import signal
import sys
from pathlib import Path
from typing import Optional

import uvicorn

from core import CameraManager, Logger, TCPServer, get_logger, load_config
from core.database import (
    init_database,
    CameraRepository
)
from core.api import create_app, set_camera_manager
from core.process_manager import ProcessManager
from core.tcp import InternalTCPRelay


class AIService:
    """Main-process orchestrator (no inference — that runs in child processes)."""

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.config_path = config_path
        self.config = None
        self.logger: Optional[Logger] = None

        # Servers
        self.tcp_server: Optional[TCPServer] = None
        self.relay_server: Optional[InternalTCPRelay] = None

        # Repositories (used by the HTTP API and CameraManager)
        self.camera_repository: Optional[CameraRepository] = None


        # High-level managers
        self.camera_manager: Optional[CameraManager] = None
        self.process_manager: Optional[ProcessManager] = None

        self._running = False

    async def initialize(self) -> bool:
        try:
            self.config = load_config(self.config_path)

            log_config = self.config.get_logging_config()
            Logger.initialize(**log_config)
            self.logger = get_logger()

            self.logger.info("=" * 80)
            self.logger.info("Ship Intelligent Monitoring System - AI Service (main process)")
            self.logger.info("=" * 80)

            tcp_config = self.config.get_tcp_config()
            # Qt broadcast server
            self.tcp_server = TCPServer(
                host=tcp_config.get("host", "0.0.0.0"),
                port=tcp_config.get("port", 9002),
                heartbeat_interval=tcp_config.get("heartbeat_interval", 5),
            )
            await self.tcp_server.start()

            # Internal relay — must be listening before workers connect
            relay_host = tcp_config.get("host", "0.0.0.0")
            relay_port = tcp_config.get("internal_port", 9003)
            self.relay_server = InternalTCPRelay(
                host=relay_host,
                port=relay_port,
                qt_server=self.tcp_server,
            )
            await self.relay_server.start()


            # Database
            db_config = self.config.get("database", {})
            db_path = db_config.get("path", "data/cameras.db")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            init_database(db_path)

            # Camera config manager (no RTSP decoding in main process)
            self.camera_repository = CameraRepository()
            self.camera_manager = CameraManager(
                camera_repository=self.camera_repository,
            )
            await self.camera_manager.start()


            # Process manager
            relay_connect_host = tcp_config.get("internal_connect_host", "127.0.0.1")
            self.process_manager = ProcessManager(
                config_path=self.config_path,
                relay_host=relay_connect_host,
                relay_port=relay_port,
            )
            self.camera_manager.set_process_manager(self.process_manager)

            self.logger.info("Main process initialized — inference workers will start next")
            return True

        except Exception as exc:
            msg = f"Failed to initialize AI Service: {exc}"
            if self.logger:
                self.logger.error(msg)
            else:
                print(msg)
            return False

    async def start(self) -> None:
        if self._running:
            self.logger.warning("AI Service already running")
            return
        self._running = True
        await self.process_manager.start_all(self.camera_repository)
        self.logger.info("AI Service started")

    async def stop(self) -> None:
        if not self._running:
            return
        self.logger.info("Stopping AI Service…")
        self._running = False

        if self.process_manager:
            await self.process_manager.stop_all()
        if self.relay_server:
            await self.relay_server.stop()
        if self.tcp_server:
            await self.tcp_server.stop()

        self.logger.info("AI Service stopped")


async def main() -> None:
    service = AIService()

    if not await service.initialize():
        print("Failed to initialize AI Service")
        sys.exit(1)

    def _signal_handler(sig, frame):
        print("\nShutdown signal received…")
        asyncio.create_task(service.stop())

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    api_config = service.config.get("api", {})
    cors_origins = api_config.get("cors_origins", ["http://localhost", "http://127.0.0.1"])
    app = create_app(cors_origins)
    set_camera_manager(service.camera_manager)

    uvicorn_config = uvicorn.Config(
        app,
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 9001),
        log_level="info",
        workers=1,  # must be 1 — workers > 1 would fork the process manager
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)

    service.logger.info(
        f"Starting HTTP API on {api_config.get('host', '0.0.0.0')}:{api_config.get('port', 9001)}"
    )
    service.logger.info(
        f"API docs: http://127.0.0.1:{api_config.get('port', 9001)}/docs"
    )

    await service.start()

    try:
        await asyncio.gather(
            uvicorn_server.serve(),
            _keep_running(service),
        )
    except KeyboardInterrupt:
        pass
    finally:
        await service.stop()


async def _keep_running(service: AIService) -> None:
    while service._running:
        await asyncio.sleep(1)


if __name__ == "__main__":
    # On Linux the default start method is 'fork', which is unsafe when asyncio
    # and SQLAlchemy connections are already live.  Spawn is safe on all platforms.
    multiprocessing.set_start_method("spawn", force=True)

    for d in ("config", "logs", "models", "data", "data/screenshots"):
        Path(d).mkdir(parents=True, exist_ok=True)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAI Service terminated by user")
    except Exception as exc:
        print(f"Fatal error: {exc}")
        sys.exit(1)
