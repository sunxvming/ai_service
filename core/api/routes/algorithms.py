"""
HTTP routes for algorithm (model config) CRUD.
"""

from fastapi import APIRouter, HTTPException

from ...database import AlgorithmRepository
from ..schemas import (
    AlgorithmCreate,
    AlgorithmResponse,
    AlgorithmUpdate,
    error_response,
    success_response,
)

router = APIRouter(tags=["algorithms"])
_repo = AlgorithmRepository()


@router.post("/algorithms")
async def create_algorithm(body: AlgorithmCreate):
    existing = await _repo.get_by_name_en(body.name_en)
    if existing:
        return error_response(f"算法 '{body.name_en}' 已存在", code=409)
    data = body.model_dump()
    created = await _repo.create(data)
    return success_response(created, "创建成功")


@router.get("/algorithms")
async def list_algorithms(enabled_only: bool = False):
    items = await _repo.get_all(enabled_only=enabled_only)
    return success_response(items)


@router.get("/algorithms/{algorithm_id}")
async def get_algorithm(algorithm_id: int):
    item = await _repo.get_by_id(algorithm_id)
    if not item:
        raise HTTPException(status_code=404, detail="算法不存在")
    return success_response(item)


@router.put("/algorithms/{algorithm_id}")
async def update_algorithm(algorithm_id: int, body: AlgorithmUpdate):
    update_data = body.model_dump(exclude_none=True)
    if not update_data:
        return error_response("没有提供更新字段")
    updated = await _repo.update(algorithm_id, update_data)
    if not updated:
        raise HTTPException(status_code=404, detail="算法不存在")
    return success_response(updated, "更新成功")


@router.delete("/algorithms/{algorithm_id}")
async def delete_algorithm(algorithm_id: int):
    deleted = await _repo.delete(algorithm_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="算法不存在")
    return success_response(None, "删除成功")
