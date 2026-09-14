"""HRCS mesh packet delivery, replay suppression, and bounded forwarding."""

from __future__ import annotations

import queue
from threading import Lock
from typing import Dict, Optional, Set

from ..core.packet import HRCSPacket


class MeshNetwork:
    MAX_HOPS = 10
    PACKET_LIFETIME = 300
    BROADCAST = 0xFFFFFFFFFFFFFFFF

    def __init__(self, node_id: int):
        self.node_id = node_id
        self.rx_queue = queue.Queue()
        self.tx_queue = queue.Queue()
        self.seen_packets: Dict[int, Set[int]] = {}
        self.lock = Lock()

    def _is_seen(self, packet: HRCSPacket) -> bool:
        with self.lock:
            return packet.sequence in self.seen_packets.get(packet.source, set())

    def _mark_seen(self, packet: HRCSPacket) -> None:
        with self.lock:
            self.seen_packets.setdefault(packet.source, set()).add(packet.sequence)

    def should_forward(self, packet: HRCSPacket) -> bool:
        if packet.hop_count >= self.MAX_HOPS:
            return False
        if self._is_seen(packet):
            return False
        if packet.dest == self.node_id:
            return False
        return True

    def process_received_packet(self, packet: HRCSPacket) -> None:
        if self._is_seen(packet):
            return
        deliver_here = packet.dest in (self.node_id, self.BROADCAST)
        forward = packet.dest != self.node_id and packet.hop_count < self.MAX_HOPS
        self._mark_seen(packet)

        if deliver_here:
            self.rx_queue.put(packet)
        if forward:
            packet.hop_count += 1
            self.tx_queue.put(packet)

    def get_received_packet(self, timeout: Optional[float] = None) -> Optional[HRCSPacket]:
        try:
            return self.rx_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def queue_transmit(self, packet: HRCSPacket) -> None:
        self.tx_queue.put(packet)

    def get_next_transmit(self, timeout: Optional[float] = None) -> Optional[HRCSPacket]:
        try:
            return self.tx_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def cleanup_old_entries(self) -> None:
        """Compatibility hook; timestamped replay-window pruning is not yet implemented."""
        return None
