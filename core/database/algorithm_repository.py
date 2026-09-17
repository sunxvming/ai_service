"""
Algorithm repository for CRUD operations on the algorithms table.
"""

import json
from typing import List, Optional

from sqlalchemy import select

from .connection import get_session
from .models import AlgorithmModel


class AlgorithmRepository:
    """Repository for Algorithm CRUD operations."""

    async def create(self, data: dict) -> dict:
        async with get_session() as session:
            model = AlgorithmModel(
                name_cn=data["name_cn"],
                name_en=data["name_en"],
                description=data.get("description"),
                model_path=data["model_path"],
                bmodel_path=data.get("bmodel_path"),
                bmodel_class_names=json.dumps(data["bmodel_class_names"], ensure_ascii=False) if data.get("bmodel_class_names") else None,
                confidence_threshold=data.get("confidence_threshold", 0.5),
                label_map=json.dumps(data["label_map"], ensure_ascii=False),
                risk_level=data.get("risk_level", "中"),
                alarm_cooldown=data.get("alarm_cooldown", 30),
                alarm_window=data.get("alarm_window", 15),
                alarm_threshold=data.get("alarm_threshold", 15),
                voice_text=data.get("voice_text"),
                enabled=data.get("enabled", 1),
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return self._to_dict(model)

    async def get_by_id(self, algorithm_id: int) -> Optional[dict]:
        async with get_session() as session:
            model = await session.get(AlgorithmModel, algorithm_id)
            return self._to_dict(model) if model else None

    async def get_by_name_en(self, name_en: str) -> Optional[dict]:
        async with get_session() as session:
            result = await session.execute(
                select(AlgorithmModel).where(AlgorithmModel.name_en == name_en)
            )
            model = result.scalar_one_or_none()
            return self._to_dict(model) if model else None

    async def get_all(self, enabled_only: bool = False) -> List[dict]:
        async with get_session() as session:
            query = select(AlgorithmModel).order_by(AlgorithmModel.id)
            if enabled_only:
                query = query.where(AlgorithmModel.enabled == 1)
            result = await session.execute(query)
            return [self._to_dict(m) for m in result.scalars().all()]

    async def update(self, algorithm_id: int, data: dict) -> Optional[dict]:
        async with get_session() as session:
            model = await session.get(AlgorithmModel, algorithm_id)
            if not model:
                return None

            for field in (
                "name_cn", "description", "model_path", "bmodel_path",
                "confidence_threshold", "risk_level", "alarm_cooldown",
                "alarm_window", "alarm_threshold", "voice_text", "enabled",
            ):
                if field in data:
                    setattr(model, field, data[field])

            if "label_map" in data:
                lm = data["label_map"]
                model.label_map = json.dumps(lm, ensure_ascii=False) if isinstance(lm, dict) else lm

            if "bmodel_class_names" in data:
                bcn = data["bmodel_class_names"]
                model.bmodel_class_names = json.dumps(bcn, ensure_ascii=False) if isinstance(bcn, dict) else bcn

            await session.commit()
            await session.refresh(model)
            return self._to_dict(model)

    async def delete(self, algorithm_id: int) -> bool:
        async with get_session() as session:
            model = await session.get(AlgorithmModel, algorithm_id)
            if not model:
                return False
            await session.delete(model)
            await session.commit()
            return True

    async def count(self) -> int:
        async with get_session() as session:
            result = await session.execute(select(AlgorithmModel))
            return len(result.scalars().all())

    def _to_dict(self, model: AlgorithmModel) -> dict:
        label_map = model.label_map
        if isinstance(label_map, str):
            try:
                label_map = json.loads(label_map)
            except (json.JSONDecodeError, TypeError):
                label_map = {}
        bmodel_class_names = model.bmodel_class_names
        if isinstance(bmodel_class_names, str):
            try:
                bmodel_class_names = json.loads(bmodel_class_names)
            except (json.JSONDecodeError, TypeError):
                bmodel_class_names = {}
        return {
            "id": model.id,
            "name_cn": model.name_cn,
            "name_en": model.name_en,
            "description": model.description,
            "model_path": model.model_path,
            "bmodel_path": model.bmodel_path,
            "bmodel_class_names": bmodel_class_names,
            "confidence_threshold": model.confidence_threshold,
            "label_map": label_map,
            "risk_level": model.risk_level,
            "alarm_cooldown": model.alarm_cooldown,
            "alarm_window": model.alarm_window,
            "alarm_threshold": model.alarm_threshold,
            "voice_text": model.voice_text,
            "enabled": model.enabled,
        }
