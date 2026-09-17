"""
Camera manager: DB-backed camera configuration and runtime CRUD.

In the multi-process architecture this class no longer owns RTSP decoders.
It is responsible only for camera configuration persistence (via the
repository) and for coordinating worker process lifecycle through the
injected ProcessManager.
"""

from typing import Dict, List, Optional, TYPE_CHECKING

from ..models import CameraConfig, CameraStatus
from ..utils import get_logger

if TYPE_CHECKING:
    from ..process_manager import ProcessManager


class CameraManager:
    """
    Manages camera configurations stored in the database.
    Delegates RTSP decoding and inference to per-camera child processes
    managed by ProcessManager.

    Call set_process_manager() after both objects are constructed so that
    add/update/remove operations automatically spawn/restart/stop the
    corresponding worker processes.
    """

    def __init__(self, camera_repository) -> None:
        self.camera_repository = camera_repository
        self.cameras: Dict[str, CameraConfig] = {}
        self._process_manager: Optional["ProcessManager"] = None
        self.logger = get_logger()

    def set_process_manager(self, pm: "ProcessManager") -> None:
        """Inject the ProcessManager after both objects are constructed."""
        self._process_manager = pm

    async def start(self) -> None:
        """Load camera configurations from DB into the local cache."""
        all_cameras = await self.camera_repository.get_all()
        self.cameras = {cam.id: cam for cam in all_cameras if cam.enabled}
        self.logger.info(f"CameraManager: loaded {len(self.cameras)} enabled camera(s) from DB")

    async def stop(self) -> None:
        """No-op — worker processes are managed by ProcessManager."""
        pass

    # ------------------------------------------------------------------
    # CRUD — each mutates the DB and updates the worker process
    # ------------------------------------------------------------------

    async def add_camera(self, camera: CameraConfig) -> CameraConfig:
        """Persist a new camera and spawn its inference worker."""
        created = await self.camera_repository.create(camera)

        if created.id in self.cameras:
            raise ValueError(f"Camera '{created.id}' already exists in cache")

        self.cameras[created.id] = created
        self.logger.info(f"CameraManager: added camera {created.id} ({created.ipc_name})")

        if created.enabled and self._process_manager and created.ipc_id is not None:
            self._process_manager.spawn_worker(created.ipc_id)

        return created

    async def update_camera(self, camera: CameraConfig) -> CameraConfig:
        """Persist camera changes; restart the worker if RTSP URL or enabled state changed."""
        if camera.id not in self.cameras:
            raise ValueError(f"Camera '{camera.id}' not found")

        old = self.cameras[camera.id]
        updated = await self.camera_repository.update(camera)
        self.cameras[camera.id] = updated

        if self._process_manager and updated.ipc_id is not None:
            rtsp_changed = old.rtsp_url != updated.rtsp_url
            enabled_changed = old.enabled != updated.enabled

            if rtsp_changed or enabled_changed:
                if updated.enabled:
                    self._process_manager.restart_worker(updated.ipc_id)
                else:
                    self._process_manager.stop_worker(updated.ipc_id)
            elif updated.enabled:
                # Algorithm IDs or other config changed — restart to reload
                self._process_manager.restart_worker(updated.ipc_id)

        self.logger.info(f"CameraManager: updated camera {camera.id}")
        return updated

    async def remove_camera(self, camera_id: str) -> bool:
        """Delete a camera from the DB and terminate its worker."""
        ipc_id = int(camera_id) if camera_id.isdigit() else 0

        if not await self.camera_repository.delete(ipc_id):
            return False

        self.cameras.pop(camera_id, None)

        if self._process_manager and ipc_id:
            self._process_manager.stop_worker(ipc_id)

        self.logger.info(f"CameraManager: removed camera {camera_id}")
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def get_camera_config(self, camera_id: str) -> Optional[CameraConfig]:
        return await self.camera_repository.get_by_id(camera_id)

    async def get_all_camera_configs(self) -> List[CameraConfig]:
        return await self.camera_repository.get_all()

    def get_camera_name(self, camera_id: str) -> str:
        camera = self.cameras.get(camera_id)
        return camera.name if camera else camera_id

    def get_camera_status(self, camera_id: str) -> Optional[CameraStatus]:
        """Return a lightweight status based on whether the worker process is alive."""
        if self._process_manager is None:
            return CameraStatus(camera_id=camera_id, online=False)
        try:
            ipc_id = int(camera_id)
        except (ValueError, TypeError):
            return CameraStatus(camera_id=camera_id, online=False)
        online = self._process_manager.is_worker_alive(ipc_id)
        return CameraStatus(camera_id=camera_id, online=online, fps=0.0)

    def get_online_cameras(self) -> List[str]:
        """Return IDs of cameras whose worker processes are currently alive."""
        if self._process_manager is None:
            return []
        return [str(ipc_id) for ipc_id in self._process_manager.get_alive_workers()]
