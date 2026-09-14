"""Deterministic in-memory HRCS transport for software end-to-end tests."""

from __future__ import annotations

import queue
import threading
from typing import Dict, Optional

from .base import BaseModem


class SimulatedLink:
    """Shared broadcast medium whose endpoints exchange raw modem bytes."""

    def __init__(self):
        self._queues: Dict[int, queue.Queue[bytes]] = {}
        self._lock = threading.Lock()

    def endpoint(self, node_id: int) -> "SimulatedModem":
        with self._lock:
            if node_id in self._queues:
                raise ValueError(f"Simulated endpoint {node_id} already exists")
            self._queues[node_id] = queue.Queue()
        return SimulatedModem(self, node_id)

    def _transmit(self, source_id: int, data: bytes) -> bool:
        with self._lock:
            targets = [q for node_id, q in self._queues.items() if node_id != source_id]
        for target in targets:
            target.put(bytes(data))
        return bool(targets)

    def _receive(self, node_id: int, timeout: float) -> Optional[bytes]:
        with self._lock:
            inbound = self._queues[node_id]
        try:
            return inbound.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None


class SimulatedModem(BaseModem):
    def __init__(self, link: SimulatedLink, node_id: int):
        super().__init__(enabled=True)
        self.link = link
        self.node_id = node_id
        self.band_name = "simulated"

    def transmit(self, data: bytes) -> bool:
        return self.link._transmit(self.node_id, data)

    def receive(self, timeout: float = 1.0) -> Optional[bytes]:
        return self.link._receive(self.node_id, timeout)

    def is_available(self) -> bool:
        return True
