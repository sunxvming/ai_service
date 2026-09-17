"""
HTTP routes for alarm event CRUD with filtering and status management.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ...database import AlarmEventRepository
from ..schemas import AlarmEventStatusUpdate, error_response, success_response

router = APIRouter(tags=["alarm_events"])
_repo = AlarmEventRepository()


@router.get("/alarm-events")
async def list_alarm_events(
    camera_id: Optional[int] = Query(None, description="摄像头ID筛选"),
    algorithm_id: Optional[int] = Query(None, description="算法ID筛选"),
    status: Optional[str] = Query(None, description="状态筛选: 未处理/已查看/已处理"),
    start_time: Optional[str] = Query(None, description="开始时间 ISO8601"),
    end_time: Optional[str] = Query(None, description="结束时间 ISO8601"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
):
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None
    try:
        if start_time:
            start_dt = datetime.fromisoformat(start_time)
        if end_time:
            end_dt = datetime.fromisoformat(end_time)
    except ValueError as e:
        return error_response(f"时间格式错误: {e}")

    result = await _repo.get_list(
        camera_id=camera_id,
        algorithm_id=algorithm_id,
        status=status,
        start_time=start_dt,
        end_time=end_dt,
        page=page,
        page_size=page_size,
    )
    return success_response(result)


@router.get("/alarm-events/{event_id}")
async def get_alarm_event(event_id: int):
    item = await _repo.get_by_id(event_id)
    if not item:
        raise HTTPException(status_code=404, detail="报警事件不存在")
    return success_response(item)


@router.put("/alarm-events/{event_id}/status")
async def update_alarm_event_status(event_id: int, body: AlarmEventStatusUpdate):
    updated = await _repo.update_status(event_id, body.status)
    if not updated:
        raise HTTPException(status_code=404, detail="报警事件不存在")
    return success_response(updated, "状态更新成功")


@router.delete("/alarm-events/{event_id}")
async def delete_alarm_event(event_id: int):
    deleted = await _repo.delete(event_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="报警事件不存在")
    return success_response(None, "删除成功")
