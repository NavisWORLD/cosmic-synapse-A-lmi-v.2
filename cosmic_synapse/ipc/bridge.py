"""Versioned IPC bridge between A-LMI and Cosmic Synapse/Unity.

Message construction/validation is transport-independent. The optional
``websockets`` dependency is loaded only when the server is actually run.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional

from .schema import decode_message, encode_message


class IPCBridge:
    """Bidirectional v1 JSON bridge with an optional WebSocket transport."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = int(port)
        self.clients: set[Any] = set()
        self.on_command_received: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_status_update: Optional[Callable[[Dict[str, Any]], None]] = None

    async def register_client(self, websocket: Any) -> None:
        self.clients.add(websocket)

    async def unregister_client(self, websocket: Any) -> None:
        self.clients.discard(websocket)

    async def handle_client(self, websocket: Any, *_args: Any) -> None:
        await self.register_client(websocket)
        try:
            async for raw in websocket:
                await self.process_raw_message(websocket, raw)
        except Exception as exc:
            # Connection shutdown/errors are transport concerns. Keep the core
            # parser strict but do not crash the server loop on a disconnected client.
            self.logger.debug("IPC client loop ended: %s", exc)
        finally:
            await self.unregister_client(websocket)

    async def process_raw_message(self, websocket: Any, raw: str | bytes) -> None:
        """Validate a serialized v1 message before dispatch."""

        message = decode_message(raw)
        await self.process_message(websocket, message)

    async def process_message(self, websocket: Any, message: Dict[str, Any]) -> None:
        """Dispatch an already-decoded v1 message."""

        # Re-encode/decode to validate callers that bypass process_raw_message.
        validated = decode_message(
            encode_message(message["type"], message.get("payload", {}))
        )
        msg_type = validated["type"]
        payload = validated["payload"]

        if msg_type == "command":
            if self.on_command_received:
                self.on_command_received(payload)
            await websocket.send(
                encode_message(
                    "command_received",
                    {"command_id": payload.get("id")},
                )
            )
            return

        if msg_type == "status":
            if self.on_status_update:
                self.on_status_update(payload)
            return

        if msg_type == "pattern_data":
            self.logger.info("Received pattern data: %s", payload.get("pattern_type"))
            return

        if msg_type == "command_received":
            self.logger.debug("Received command acknowledgement: %s", payload.get("command_id"))
            return

        raise ValueError(f"Unsupported IPC message type: {msg_type}")

    async def send_to_all(self, raw_message: str) -> None:
        """Broadcast one already-validated serialized message."""

        decode_message(raw_message)
        if not self.clients:
            return
        await asyncio.gather(
            *(client.send(raw_message) for client in tuple(self.clients)),
            return_exceptions=True,
        )

    def build_spawn_command(
        self,
        mass_type: str,
        position: tuple[float, ...],
        properties: Dict[str, Any],
        *,
        command_id: str | None = None,
    ) -> str:
        """Build a v1 spawn command without requiring an event loop/transport."""

        if not mass_type:
            raise ValueError("mass_type must be non-empty")
        if len(position) not in {2, 3}:
            raise ValueError("position must contain two or three coordinates")
        command_id = command_id or f"spawn_{time.time_ns()}"
        return encode_message(
            "command",
            {
                "command": "spawn_mass",
                "id": command_id,
                "mass_type": mass_type,
                "position": [float(value) for value in position],
                "properties": dict(properties),
            },
        )

    async def broadcast_spawn_command(
        self,
        mass_type: str,
        position: tuple[float, ...],
        properties: Dict[str, Any],
        *,
        command_id: str | None = None,
    ) -> str:
        """Build and broadcast a spawn command from async code."""

        raw = self.build_spawn_command(
            mass_type, position, properties, command_id=command_id
        )
        await self.send_to_all(raw)
        return raw

    def send_spawn_command(
        self,
        mass_type: str,
        position: tuple[float, ...],
        properties: Dict[str, Any],
        *,
        command_id: str | None = None,
    ) -> str:
        """Compatibility helper: build a command and schedule only if a loop exists.

        Synchronous callers always receive the serialized command. If called
        inside an active asyncio loop the command is also scheduled for
        broadcast. Outside a loop it does not pretend asynchronous delivery
        occurred; callers can pass the returned message to ``send_to_all`` or
        use ``broadcast_spawn_command``.
        """

        raw = self.build_spawn_command(
            mass_type, position, properties, command_id=command_id
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return raw
        loop.create_task(self.send_to_all(raw))
        return raw

    async def run(self) -> None:
        """Run the optional WebSocket server until cancelled."""

        try:
            from websockets.asyncio.server import serve
        except ImportError:
            try:
                from websockets.server import serve
            except ImportError as exc:
                raise RuntimeError(
                    "WebSocket IPC transport is optional; install the ipc extra"
                ) from exc

        self.logger.info("Starting IPC bridge on ws://%s:%s", self.host, self.port)
        async with serve(self.handle_client, self.host, self.port):
            await asyncio.Future()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    bridge = IPCBridge()
    try:
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
