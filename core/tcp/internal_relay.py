"""
Internal TCP relay server (port 9003).

Accepts connections from inference worker child processes, reads newline-
delimited JSON messages, and forwards them verbatim to the Qt-facing
TCPServer on port 9002.  The protocol is identical to the existing Qt
broadcast format so no message schema changes are required.
"""

import asyncio
import json
from typing import Optional, TYPE_CHECKING

from ..utils import get_logger

if TYPE_CHECKING:
    from .tcp_server import TCPServer


class InternalTCPRelay:
    """
    Listens on an internal port for inference worker connections.
    Every line received from a worker is parsed as JSON and re-broadcast
    to all Qt clients via the existing TCPServer instance.
    """

    def __init__(self, host: str, port: int, qt_server: "TCPServer") -> None:
        self.host = host
        self.port = port
        self._qt_server = qt_server
        self._server: Optional[asyncio.Server] = None
        self._running = False
        self.logger = get_logger()

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_worker,
            self.host,
            self.port,
        )
        self._running = True
        addr = self._server.sockets[0].getsockname() if self._server.sockets else (self.host, self.port)
        self.logger.info(f"Internal TCP relay started on {addr[0]}:{addr[1]}")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        self.logger.info("Internal TCP relay stopped")

    async def _handle_worker(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        addr = writer.get_extra_info("peername")
        self.logger.info(f"Inference worker connected to relay: {addr}")
        try:
            while self._running:
                line = await reader.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode("utf-8"))
                    await self._qt_server._broadcast_message(msg)
                except json.JSONDecodeError as exc:
                    self.logger.warning(f"Relay: invalid JSON from {addr}: {exc}")
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.logger.error(f"Relay handler error for {addr}: {exc}")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            self.logger.info(f"Inference worker disconnected from relay: {addr}")
