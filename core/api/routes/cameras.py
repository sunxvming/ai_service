"""
IpcInfo CRUD API endpoints.
Returns { code: 0, message: "...", data: {...} } format for video_system client.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..dependencies import get_camera_manager
from ..schemas import (
    IpcInfoCreate, IpcInfoUpdate, IpcPositionUpdate,
    success_response, error_response
)
from ...models import IpcInfo

router = APIRouter()


def _ipc_to_dict(ipc: IpcInfo, camera_manager) -> dict:
    """Convert IpcInfo to dict with runtime status, mask password."""
    status_info = camera_manager.get_camera_status(ipc.id)
    d = ipc.to_dict()
    d["online"] = status_info.online if status_info else False
    d["fps"] = round(status_info.fps, 2) if status_info else 0.0
    d.pop("user_pwd", None)  # Mask password
    return d


@router.post("/ipc", tags=["IpcInfo"], summary="创建摄像头")
async def create_ipc(ipc: IpcInfoCreate):
    camera_manager = get_camera_manager()
    new_ipc = IpcInfo(
        ipc_name=ipc.ipc_name,
        nvr_name=ipc.nvr_name,
        ipc_type=ipc.ipc_type,
        onvif_addr=ipc.onvif_addr,
        profile_token=ipc.profile_token,
        video_source=ipc.video_source,
        rtsp_main=ipc.rtsp_main or "",
        rtsp_sub=ipc.rtsp_sub,
        ipc_position=ipc.ipc_position,
        ipc_image=ipc.ipc_image,
        ipc_x=ipc.ipc_x,
        ipc_y=ipc.ipc_y,
        user_name=ipc.user_name,
        user_pwd=ipc.user_pwd,
        ipc_enable=ipc.ipc_enable or "启用",
        ipc_mark=ipc.ipc_mark,
    )
    try:
        created = await camera_manager.add_camera(new_ipc)
        return JSONResponse(
            status_code=201,
            content=success_response(_ipc_to_dict(created, camera_manager), "创建成功")
        )
    except ValueError as e:
        return JSONResponse(status_code=409, content=error_response(str(e), 409))
    except Exception as e:
        return JSONResponse(status_code=500, content=error_response(str(e), 500))


@router.get("/ipc", tags=["IpcInfo"], summary="获取摄像头列表")
async def list_ipc(enabled_only: bool = False):
    camera_manager = get_camera_manager()
    ipcs = await camera_manager.get_all_camera_configs()
    if enabled_only:
        ipcs = [c for c in ipcs if c.enabled]
    items = [_ipc_to_dict(c, camera_manager) for c in ipcs]
    return success_response({"list": items, "total": len(items)})


@router.get("/ipc/{ipc_id}", tags=["IpcInfo"], summary="获取单个摄像头")
async def get_ipc(ipc_id: int):
    camera_manager = get_camera_manager()
    ipc = await camera_manager.get_camera_config(str(ipc_id))
    if not ipc:
        return JSONResponse(status_code=404,
                            content=error_response(f"摄像头 {ipc_id} 不存在", 404))
    return success_response(_ipc_to_dict(ipc, camera_manager))


@router.put("/ipc/{ipc_id}", tags=["IpcInfo"], summary="更新摄像头")
async def update_ipc(ipc_id: int, ipc_update: IpcInfoUpdate):
    camera_manager = get_camera_manager()
    existing = await camera_manager.get_camera_config(str(ipc_id))
    if not existing:
        return JSONResponse(status_code=404,
                            content=error_response(f"摄像头 {ipc_id} 不存在", 404))

    update_data = ipc_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(existing, field, value)

    try:
        updated = await camera_manager.update_camera(existing)
        return success_response(_ipc_to_dict(updated, camera_manager), "更新成功")
    except Exception as e:
        return JSONResponse(status_code=500, content=error_response(str(e), 500))


@router.delete("/ipc/{ipc_id}", tags=["IpcInfo"], summary="删除摄像头")
async def delete_ipc(ipc_id: int):
    camera_manager = get_camera_manager()
    success = await camera_manager.remove_camera(str(ipc_id))
    if not success:
        return JSONResponse(status_code=404,
                            content=error_response(f"摄像头 {ipc_id} 不存在", 404))
    return success_response(message="删除成功")


@router.put("/ipc/{ipc_id}/position", tags=["IpcInfo"], summary="更新摄像头位置")
async def update_ipc_position(ipc_id: int, pos: IpcPositionUpdate):
    camera_manager = get_camera_manager()
    if camera_manager.camera_repository:
        ok = await camera_manager.camera_repository.update_position(
            ipc_id,
            ipc_position=pos.ipc_position,
            ipc_x=pos.ipc_x,
            ipc_y=pos.ipc_y,
        )
        if not ok:
            return JSONResponse(status_code=404,
                                content=error_response(f"摄像头 {ipc_id} 不存在", 404))
        ipc = await camera_manager.get_camera_config(str(ipc_id))
        return success_response(_ipc_to_dict(ipc, camera_manager) if ipc else {}, "位置更新成功")
    return JSONResponse(status_code=400, content=error_response("数据库未配置", 400))


@router.get("/health", tags=["System"], summary="健康检查")
async def health_check():
    camera_manager = get_camera_manager()
    online = len(camera_manager.get_online_cameras())
    total = len(camera_manager.cameras)
    return success_response({
        "status": "running",
        "cameras_loaded": total,
        "cameras_online": online,
    })
