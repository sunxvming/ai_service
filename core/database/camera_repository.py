"""
IpcInfo repository for database CRUD operations.
"""

from typing import List, Optional
from sqlalchemy import select

from ..models import IpcInfo
from .connection import get_session
from .models import IpcInfoModel


class IpcRepository:
    """Repository for IpcInfo CRUD operations."""

    async def create(self, ipc: IpcInfo) -> IpcInfo:
        async with get_session() as session:
            model = IpcInfoModel(
                IpcName=ipc.ipc_name,
                NvrName=ipc.nvr_name,
                IpcType=ipc.ipc_type,
                OnvifAddr=ipc.onvif_addr,
                ProfileToken=ipc.profile_token,
                VideoSource=ipc.video_source,
                RtspMain=ipc.rtsp_main,
                RtspSub=ipc.rtsp_sub,
                IpcPosition=ipc.ipc_position,
                IpcImage=ipc.ipc_image,
                IpcX=ipc.ipc_x,
                IpcY=ipc.ipc_y,
                UserName=ipc.user_name,
                UserPwd=ipc.user_pwd,
                IpcEnable=ipc.ipc_enable or "启用",
                IpcMark=ipc.ipc_mark,
                scene_id=ipc.scene_id,
                algorithm_ids=ipc.algorithm_ids,
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return self._to_ipc(model)

    async def get_by_id(self, ipc_id: int) -> Optional[IpcInfo]:
        async with get_session() as session:
            model = await session.get(IpcInfoModel, ipc_id)
            return self._to_ipc(model) if model else None

    async def get_all(self, enabled_only: bool = False) -> List[IpcInfo]:
        async with get_session() as session:
            query = select(IpcInfoModel).order_by(IpcInfoModel.IpcID)
            if enabled_only:
                query = query.where(IpcInfoModel.IpcEnable == "启用")
            result = await session.execute(query)
            return [self._to_ipc(m) for m in result.scalars().all()]

    async def update(self, ipc: IpcInfo) -> IpcInfo:
        async with get_session() as session:
            model = await session.get(IpcInfoModel, ipc.ipc_id)
            if not model:
                raise ValueError(f"IpcInfo with ID '{ipc.ipc_id}' not found")

            model.IpcName      = ipc.ipc_name
            model.NvrName      = ipc.nvr_name
            model.IpcType      = ipc.ipc_type
            model.OnvifAddr    = ipc.onvif_addr
            model.ProfileToken = ipc.profile_token
            model.VideoSource  = ipc.video_source
            model.RtspMain     = ipc.rtsp_main
            model.RtspSub      = ipc.rtsp_sub
            model.IpcPosition  = ipc.ipc_position
            model.IpcImage     = ipc.ipc_image
            model.IpcX         = ipc.ipc_x
            model.IpcY         = ipc.ipc_y
            model.UserName     = ipc.user_name
            model.UserPwd      = ipc.user_pwd
            model.IpcEnable    = ipc.ipc_enable or "启用"
            model.IpcMark      = ipc.ipc_mark
            model.scene_id     = ipc.scene_id
            model.algorithm_ids = ipc.algorithm_ids

            await session.commit()
            await session.refresh(model)
            return self._to_ipc(model)

    async def delete(self, ipc_id: int) -> bool:
        async with get_session() as session:
            model = await session.get(IpcInfoModel, ipc_id)
            if not model:
                return False
            await session.delete(model)
            await session.commit()
            return True

    async def update_position(self, ipc_id: int, ipc_position: str = None,
                               ipc_x: int = None, ipc_y: int = None) -> bool:
        async with get_session() as session:
            model = await session.get(IpcInfoModel, ipc_id)
            if not model:
                return False
            if ipc_position is not None:
                model.IpcPosition = ipc_position
            if ipc_x is not None:
                model.IpcX = ipc_x
            if ipc_y is not None:
                model.IpcY = ipc_y
            await session.commit()
            return True

    def _to_ipc(self, model: IpcInfoModel) -> IpcInfo:
        return IpcInfo(
            ipc_id=model.IpcID,
            ipc_name=model.IpcName or "",
            nvr_name=model.NvrName,
            ipc_type=model.IpcType,
            onvif_addr=model.OnvifAddr,
            profile_token=model.ProfileToken,
            video_source=model.VideoSource,
            rtsp_main=model.RtspMain or "",
            rtsp_sub=model.RtspSub,
            ipc_position=model.IpcPosition,
            ipc_image=model.IpcImage,
            ipc_x=model.IpcX,
            ipc_y=model.IpcY,
            user_name=model.UserName,
            user_pwd=model.UserPwd,
            ipc_enable=model.IpcEnable or "启用",
            ipc_mark=model.IpcMark,
            scene_id=model.scene_id,
            algorithm_ids=model.algorithm_ids,
        )


# Backward compatibility alias
CameraRepository = IpcRepository

