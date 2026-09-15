"""Layered HRCS node implementation with injectable physical transport."""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional

from .application.messaging import Messaging
from .core.crypto import HRCSSecurity
from .core.packet import HRCSPacket
from .network.discovery import NeighborDiscovery
from .network.mesh import MeshNetwork
from .network.routing import GoldenRatioRouter
from .physical.acoustic import AcousticModem
from .physical.base import BaseModem
from .physical.radio import RadioModem


class HRCSNode:
    def __init__(
        self,
        node_id: int,
        network_key: Optional[str] = None,
        acoustic_only: bool = False,
        modem: Optional[BaseModem] = None,
    ):
        self.node_id = node_id
        self.running = False
        self.security = HRCSSecurity(pre_shared_key=network_key)
        self.modems: Dict[str, BaseModem] = {}

        if modem is not None:
            self.modems[modem.band_name] = modem
            self.active_modem = modem
        else:
            if not acoustic_only:
                radio_modem = RadioModem()
                if radio_modem.is_available():
                    self.modems["radio"] = radio_modem
                    self.active_modem = radio_modem
                else:
                    acoustic_only = True
            if acoustic_only:
                acoustic_modem = AcousticModem()
                self.modems["acoustic"] = acoustic_modem
                self.active_modem = acoustic_modem

        self.router = GoldenRatioRouter(node_id)
        self.mesh = MeshNetwork(node_id)
        self.discovery = NeighborDiscovery(node_id)
        self.messaging = Messaging(node_id, self.mesh)
        self.tx_thread: Optional[threading.Thread] = None
        self.rx_thread: Optional[threading.Thread] = None
        # TX and RX are independent operations for the modem abstraction. A
        # single shared lock caused a waiting receive to starve queued sends in
        # deterministic transports.
        self.tx_lock = threading.Lock()
        self.rx_lock = threading.Lock()

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.tx_thread = threading.Thread(target=self._tx_worker, daemon=True)
        self.rx_thread = threading.Thread(target=self._rx_worker, daemon=True)
        self.tx_thread.start()
        self.rx_thread.start()

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        if self.tx_thread:
            self.tx_thread.join(timeout=2.0)
        if self.rx_thread:
            self.rx_thread.join(timeout=2.0)

    def send_message(self, dest_id: int, message: str) -> bool:
        return self.messaging.send_message(dest_id, message)

    def receive_message(self, timeout: Optional[float] = None):
        return self.messaging.receive_message(timeout=timeout)

    def _tx_worker(self) -> None:
        while self.running:
            try:
                packet = self.mesh.get_next_transmit(timeout=0.05)
                if packet is None:
                    continue
                encrypted = self.security.encrypt_packet(packet.serialize())
                with self.tx_lock:
                    success = self.active_modem.transmit(encrypted)
                if success:
                    logging.debug("HRCS TX %s bytes to %016X", len(encrypted), packet.dest)
            except Exception as exc:
                logging.error("HRCS TX error: %s", exc)

    def _rx_worker(self) -> None:
        while self.running:
            try:
                with self.rx_lock:
                    data = self.active_modem.receive(timeout=0.05)
                if data is None:
                    if self.discovery.should_send_hello():
                        self.mesh.queue_transmit(self.discovery.create_hello_packet())
                    continue
                try:
                    packet_bytes = self.security.decrypt_packet(data)
                    packet = HRCSPacket.deserialize(packet_bytes)
                except ValueError as exc:
                    logging.warning("Rejected HRCS frame: %s", exc)
                    continue

                if packet.type == "HELLO":
                    # Discovery is control-plane traffic, not an application
                    # message. HELLO is one-hop broadcast in the active node
                    # implementation and therefore does not enter MeshNetwork's
                    # application delivery queue.
                    self.discovery.update_neighbor(packet.source, -50, 30)
                    self.router.update_neighbor(packet.source, -50, 30)
                    continue

                self.mesh.process_received_packet(packet)
            except Exception as exc:
                logging.error("HRCS RX error: %s", exc)
