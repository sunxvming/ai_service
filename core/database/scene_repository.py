"""
AlarmScene repository for CRUD operations.
"""

from typing import List, Optional
from sqlalchemy import select
from .connection import get_session
from .models import AlarmSceneModel


class SceneRepository:
    """Repository for AlarmScene CRUD operations."""

    async def create(self, data: dict) -> dict:
        """Create a new scene."""
        async with get_session() as session:
            model = AlarmSceneModel(
                scene_name=data["scene_name"],
                scene_description=data.get("scene_description"),
                algorithm_ids=data.get("algorithm_ids", ""),
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return self._to_dict(model)

    async def get_by_id(self, scene_id: int) -> Optional[dict]:
        """Get scene by ID."""
        async with get_session() as session:
            model = await session.get(AlarmSceneModel, scene_id)
            return self._to_dict(model) if model else None

    async def get_by_name(self, scene_name: str) -> Optional[dict]:
        """Get scene by name."""
        async with get_session() as session:
            result = await session.execute(
                select(AlarmSceneModel).where(AlarmSceneModel.scene_name == scene_name)
            )
            model = result.scalar_one_or_none()
            return self._to_dict(model) if model else None

    async def get_all(self) -> List[dict]:
        """Get all scenes."""
        async with get_session() as session:
            query = select(AlarmSceneModel).order_by(AlarmSceneModel.scene_id)
            result = await session.execute(query)
            return [self._to_dict(m) for m in result.scalars().all()]

    async def update(self, scene_id: int, data: dict) -> Optional[dict]:
        """Update scene."""
        async with get_session() as session:
            model = await session.get(AlarmSceneModel, scene_id)
            if not model:
                return None

            for field in ("scene_name", "scene_description", "algorithm_ids"):
                if field in data:
                    setattr(model, field, data[field])

            await session.commit()
            await session.refresh(model)
            return self._to_dict(model)

    async def delete(self, scene_id: int) -> bool:
        """Delete scene."""
        async with get_session() as session:
            model = await session.get(AlarmSceneModel, scene_id)
            if not model:
                return False
            await session.delete(model)
            await session.commit()
            return True

    def _to_dict(self, model: AlarmSceneModel) -> dict:
        """Convert model to dict."""
        return {
            "scene_id": model.scene_id,
            "scene_name": model.scene_name,
            "scene_description": model.scene_description,
            "algorithm_ids": model.algorithm_ids,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "updated_at": model.updated_at.isoformat() if model.updated_at else None,
        }
