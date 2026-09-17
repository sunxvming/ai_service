"""
Alarm event repository for CRUD operations on the alarm_events table.
"""

import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy import select, and_
from sqlalchemy.orm import aliased

from .connection import get_session
from .models import AlarmEventModel, IpcInfoModel, AlgorithmModel


class AlarmEventRepository:
    """Repository for AlarmEvent CRUD operations."""

    async def create(self, data: dict) -> dict:
        async with get_session() as session:
            detection_info = data.get("detection_info")
            if isinstance(detection_info, dict):
                detection_info = json.dumps(detection_info, ensure_ascii=False)

            model = AlarmEventModel(
                camera_id=data["camera_id"],
                algorithm_id=data["algorithm_id"],
                alarm_time=data.get("alarm_time") or datetime.now(),
                risk_level=data.get("risk_level", "中"),
                screenshot_path=data.get("screenshot_path"),
                detection_info=detection_info,
                status=data.get("status", "未处理"),
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return self._to_dict(model)

    async def get_by_id(self, event_id: int) -> Optional[dict]:
        async with get_session() as session:
            # Join with IpcInfoModel and AlgorithmModel to get names
            query = (
                select(AlarmEventModel, IpcInfoModel.IpcName, AlgorithmModel.name_cn)
                .outerjoin(IpcInfoModel, AlarmEventModel.camera_id == IpcInfoModel.IpcID)
                .outerjoin(AlgorithmModel, AlarmEventModel.algorithm_id == AlgorithmModel.id)
                .where(AlarmEventModel.id == event_id)
            )
            result = await session.execute(query)
            row = result.first()
            if not row:
                return None

            model, camera_name, algorithm_name = row
            data = self._to_dict(model)
            data["camera_name"] = camera_name or f"Camera-{model.camera_id}"
            data["algorithm_name"] = algorithm_name or f"Algorithm-{model.algorithm_id}"
            return data

    async def get_list(
        self,
        camera_id: Optional[int] = None,
        algorithm_id: Optional[int] = None,
        status: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        async with get_session() as session:
            filters = []
            if camera_id is not None:
                filters.append(AlarmEventModel.camera_id == camera_id)
            if algorithm_id is not None:
                filters.append(AlarmEventModel.algorithm_id == algorithm_id)
            if status is not None:
                filters.append(AlarmEventModel.status == status)
            if start_time is not None:
                filters.append(AlarmEventModel.alarm_time >= start_time)
            if end_time is not None:
                filters.append(AlarmEventModel.alarm_time <= end_time)

            # Join with IpcInfoModel and AlgorithmModel
            base_query = (
                select(AlarmEventModel, IpcInfoModel.IpcName, AlgorithmModel.name_cn)
                .outerjoin(IpcInfoModel, AlarmEventModel.camera_id == IpcInfoModel.IpcID)
                .outerjoin(AlgorithmModel, AlarmEventModel.algorithm_id == AlgorithmModel.id)
            )
            if filters:
                base_query = base_query.where(and_(*filters))

            # Count total
            count_result = await session.execute(base_query)
            total = len(count_result.all())

            # Paginated query
            query = base_query.order_by(AlarmEventModel.alarm_time.desc())
            query = query.offset((page - 1) * page_size).limit(page_size)
            result = await session.execute(query)

            items = []
            for model, camera_name, algorithm_name in result.all():
                data = self._to_dict(model)
                data["camera_name"] = camera_name or f"Camera-{model.camera_id}"
                data["algorithm_name"] = algorithm_name or f"Algorithm-{model.algorithm_id}"
                items.append(data)

            total_pages = (total + page_size - 1) // page_size  # Ceiling division
            if total_pages < 1:
                total_pages = 1

            return {
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "items": items,
            }

    async def update_status(self, event_id: int, status: str) -> Optional[dict]:
        async with get_session() as session:
            model = await session.get(AlarmEventModel, event_id)
            if not model:
                return None
            model.status = status
            await session.commit()
            await session.refresh(model)
            return self._to_dict(model)

    async def delete(self, event_id: int) -> bool:
        async with get_session() as session:
            model = await session.get(AlarmEventModel, event_id)
            if not model:
                return False
            await session.delete(model)
            await session.commit()
            return True

    def _to_dict(self, model: AlarmEventModel) -> dict:
        detection_info = model.detection_info
        if isinstance(detection_info, str):
            try:
                detection_info = json.loads(detection_info)
            except (json.JSONDecodeError, TypeError):
                detection_info = None

        alarm_time = model.alarm_time
        if isinstance(alarm_time, datetime):
            alarm_time = alarm_time.isoformat()

        return {
            "id": model.id,
            "camera_id": model.camera_id,
            "algorithm_id": model.algorithm_id,
            "alarm_time": alarm_time,
            "risk_level": model.risk_level,
            "screenshot_path": model.screenshot_path,
            "detection_info": detection_info,
            "status": model.status,
        }
