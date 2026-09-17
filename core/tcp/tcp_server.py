"""
TCP Server module for broadcasting AI results to Qt clients.
Supports multiple concurrent client connections with newline-delimited JSON protocol.
"""

import asyncio
import json
from typing import Dict, List, Optional, Set

from ..models import AlarmEvent, CameraStatus, DetectionResult
from ..utils import get_logger


class TCPServer:
    """
    Asynchronous TCP server for broadcasting AI results to multiple Qt clients.
    Uses newline-delimited JSON protocol matching Qt client expectations.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9002, heartbeat_interval: int = 5):
        """
        Initialize TCP server.

        Args:
            host: Server host address
            port: Server port
            heartbeat_interval: Heartbeat interval in seconds
        """
        self.host = host
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.logger = get_logger()

        self._server: Optional[asyncio.Server] = None
        self._clients: Set[asyncio.StreamWriter] = set()
        self._lock = asyncio.Lock()
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start TCP server."""
        if self._running:
            self.logger.warning("TCP server already running")
            return

        self._server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port
        )

        addr = self._server.sockets[0].getsockname() if self._server.sockets else (self.host, self.port)
        self.logger.info(f"TCP server started on {addr[0]}:{addr[1]}")

        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """Stop TCP server and close all client connections."""
        if not self._running:
            return

        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            for writer in self._clients:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
            self._clients.clear()

        if self._server:
            self._server.close()
            await self._server.wait_closed()

        self.logger.info("TCP server stopped")

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ) -> None:
        """
        Handle new client connection.

        Args:
            reader: Stream reader
            writer: Stream writer
        """
        addr = writer.get_extra_info('peername')
        self.logger.info(f"New client connected: {addr}")

        async with self._lock:
            self._clients.add(writer)

        try:
            # Keep connection alive (client sends no data, only receives)
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Client handler error for {addr}: {e}")
        finally:
            async with self._lock:
                self._clients.discard(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            self.logger.info(f"Client disconnected: {addr}")

    async def broadcast_detection(self, detection_result: DetectionResult) -> None:
        """
        Broadcast detection result to all connected clients.

        Args:
            detection_result: Detection result to broadcast
        """
        await self._broadcast_message(detection_result.to_dict())

    async def broadcast_alarm(self, alarm_event: AlarmEvent) -> None:
        """
        Broadcast alarm event to all connected clients.

        Args:
            alarm_event: Alarm event to broadcast
        """
        await self._broadcast_message(alarm_event.to_dict())

    async def broadcast_camera_status(self, camera_status: CameraStatus) -> None:
        """
        Broadcast camera status to all connected clients.

        Args:
            camera_status: Camera status to broadcast
        """
        await self._broadcast_message(camera_status.to_dict())

    async def _broadcast_message(self, message: dict) -> None:
        """
        Broadcast JSON message to all connected clients.

        Args:
            message: Message dictionary to broadcast
        """
        if not self._clients:
            return

        json_str = json.dumps(message, ensure_ascii=False)
        data = (json_str + "\n").encode('utf-8')
        # print send data
        # print(f"===Broadcasting message, data is: {data}")

        async with self._lock:
            disconnected = []
            for writer in self._clients:
                try:
                    writer.write(data)
                    await writer.drain()
                except Exception as e:
                    self.logger.warning(f"Failed to send to client: {e}")
                    disconnected.append(writer)

            for writer in disconnected:
                self._clients.discard(writer)
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat to all clients."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                heartbeat = {
                    "type": "heartbeat",
                    "timestamp": int(asyncio.get_event_loop().time())
                }
                await self._broadcast_message(heartbeat)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")

    def get_client_count(self) -> int:
        """Get number of connected clients."""
        return len(self._clients)
