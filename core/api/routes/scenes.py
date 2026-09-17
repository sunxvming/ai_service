"""
HTTP routes for scene CRUD.
"""

from fastapi import APIRouter, HTTPException
from ...database import SceneRepository
from ..schemas import (
    SceneCreate,
    SceneResponse,
    SceneUpdate,
    error_response,
    success_response,
)

router = APIRouter(tags=["scenes"])
_repo = SceneRepository()


@router.post("/alarm_scene")
async def create_scene(body: SceneCreate):
    """Create a new scene."""
    existing = await _repo.get_by_name(body.scene_name)
    if existing:
        return error_response(f"场景 '{body.scene_name}' 已存在", code=409)
    data = body.model_dump()
    created = await _repo.create(data)
    return success_response(created, "创建成功")


@router.get("/alarm_scene")
async def list_scenes():
    """Get all scenes."""
    items = await _repo.get_all()
    return success_response({"list": items, "total": len(items)})


@router.get("/alarm_scene/{scene_id}")
async def get_scene(scene_id: int):
    """Get scene by ID."""
    item = await _repo.get_by_id(scene_id)
    if not item:
        raise HTTPException(status_code=404, detail="场景不存在")
    return success_response(item)


@router.put("/alarm_scene/{scene_id}")
async def update_scene(scene_id: int, body: SceneUpdate):
    """Update scene."""
    update_data = body.model_dump(exclude_none=True)
    if not update_data:
        return error_response("没有提供更新字段")
    updated = await _repo.update(scene_id, update_data)
    if not updated:
        raise HTTPException(status_code=404, detail="场景不存在")
    return success_response(updated, "更新成功")


@router.delete("/alarm_scene/{scene_id}")
async def delete_scene(scene_id: int):
    """Delete scene."""
    deleted = await _repo.delete(scene_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="场景不存在")
    return success_response(None, "删除成功")
