"""
Process manager: spawns, monitors, and restarts inference worker child processes.
One process per enabled camera; crashed workers are automatically restarted.
"""

import asyncio
import multiprocessing
from typing import Dict, List, Optional

from .utils import get_logger
from .worker.inference_worker import run_inference_worker


class ProcessManager:
    """
    Owns the lifecycle of all inference worker child processes.

    spawn_worker / stop_worker / restart_worker are synchronous so that API
    route handlers (running in the asyncio event loop) can call them without
    awaiting — child-process operations (start/terminate/join) are short-lived
    enough to be safe on the event loop thread.
    """

    MONITOR_INTERVAL = 10  # seconds between liveness checks

    def __init__(self, config_path: str, relay_host: str, relay_port: int) -> None:
        self._config_path = config_path
        self._relay_host = relay_host
        self._relay_port = relay_port
        self._processes: Dict[int, multiprocessing.Process] = {}
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self.logger = get_logger()

    # ------------------------------------------------------------------
    # Startup / shutdown
    # ------------------------------------------------------------------

    async def start_all(self, camera_repository) -> None:
        """Spawn one worker for every enabled camera record in the DB."""
        cameras = await camera_repository.get_all(enabled_only=True)
        for cam in cameras:
            if cam.ipc_id is not None:
                self.spawn_worker(cam.ipc_id)
        self.logger.info(f"ProcessManager: spawned {len(self._processes)} worker(s)")

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_all(self) -> None:
        """Terminate all worker processes and cancel the monitor task."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        for ipc_id in list(self._processes.keys()):
            self.stop_worker(ipc_id)
        self.logger.info("ProcessManager: all workers stopped")

    # ------------------------------------------------------------------
    # Per-worker controls (synchronous — safe to call from event loop)
    # ------------------------------------------------------------------

    def spawn_worker(self, ipc_id: int) -> None:
        """Spawn a new inference worker process for the given camera."""
        existing = self._processes.get(ipc_id)
        if existing is not None and existing.is_alive():
            self.logger.warning(f"Worker cam{ipc_id} already running (pid={existing.pid})")
            return

        p = multiprocessing.Process(
            target=run_inference_worker,
            args=(ipc_id, self._config_path, self._relay_host, self._relay_port),
            daemon=True,
            name=f"inference-cam{ipc_id}",
        )
        p.start()
        self._processes[ipc_id] = p
        self.logger.info(f"Spawned worker cam{ipc_id} (pid={p.pid})")

    def stop_worker(self, ipc_id: int) -> None:
        """Terminate the worker for the given camera and clean up."""
        p = self._processes.pop(ipc_id, None)
        if p is None:
            return
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=2)
        self.logger.info(f"Stopped worker cam{ipc_id}")

    def restart_worker(self, ipc_id: int) -> None:
        """Stop and re-spawn the worker for the given camera."""
        self.stop_worker(ipc_id)
        self.spawn_worker(ipc_id)

    def is_worker_alive(self, ipc_id: int) -> bool:
        """Return True if the worker process for the given camera is running."""
        p = self._processes.get(ipc_id)
        return p is not None and p.is_alive()

    def get_alive_workers(self) -> List[int]:
        """Return IDs of cameras whose worker processes are currently alive."""
        return [ipc_id for ipc_id, p in self._processes.items() if p.is_alive()]

    # ------------------------------------------------------------------
    # Monitor loop
    # ------------------------------------------------------------------

    async def _monitor_loop(self) -> None:
        """Restart any worker processes that have exited unexpectedly."""
        while self._running:
            try:
                await asyncio.sleep(self.MONITOR_INTERVAL)
                for ipc_id, p in list(self._processes.items()):
                    if not p.is_alive():
                        self.logger.warning(
                            f"Worker cam{ipc_id} (pid={p.pid}) died (exitcode={p.exitcode}), restarting"
                        )
                        self.spawn_worker(ipc_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"ProcessManager monitor error: {e}")
