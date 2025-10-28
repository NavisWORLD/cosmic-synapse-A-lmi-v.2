"""
Main HRCS Node Implementation
Integrates all layers for complete infrastructure-free communication
"""

import time
import threading
import queue
from typing import Optional, Dict
import logging

from .core.packet import HRCSPacket
from .core.crypto import HRCSSecurity
from .physical.base import BaseModem
from .physical.acoustic import AcousticModem
from .physical.radio import RadioModem
from .network.routing import GoldenRatioRouter
from .network.mesh import MeshNetwork
from .network.discovery import NeighborDiscovery
from .application.messaging import Messaging


class HRCSNode:
    """
    Complete HRCS node implementation
    
    Combines physical layer (acoustic/radio), network layer (routing/mesh),
    and application layer (messaging) for infrastructure-free communication.
    """
    
    def __init__(self, node_id: int, network_key: Optional[str] = None, 
                 acoustic_only: bool = False):
        """
        Initialize HRCS node
        
        Args:
            node_id: Unique node identifier (64-bit)
            network_key: Shared encryption key for network
            acoustic_only: Use only acoustic modem (skip SDR if unavailable)
        """
        self.node_id = node_id
        self.running = False
        
        # Security
        self.security = HRCSSecurity(pre_shared_key=network_key)
        
        # Physical layer - try radio first, fall back to acoustic
        self.modems: Dict[str, BaseModem] = {}
        
        if not acoustic_only:
            radio_modem = RadioModem()
            if radio_modem.is_available():
                self.modems['radio'] = radio_modem
                self.active_modem = radio_modem
            else:
                logging.warning("SDR not available, using acoustic only")
                acoustic_only = True
        
        if acoustic_only:
            acoustic_modem = AcousticModem()
            self.modems['acoustic'] = acoustic_modem
            if 'active_modem' not in dir(self):
                self.active_modem = acoustic_modem
        
        # Network layer
        self.router = GoldenRatioRouter(node_id)
        self.mesh = MeshNetwork(node_id)
        self.discovery = NeighborDiscovery(node_id)
        
        # Application layer
        self.messaging = Messaging(node_id, self.mesh)
        
        # Worker threads
        self.tx_thread: Optional[threading.Thread] = None
        self.rx_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
    
    def start(self):
        """Start node operation"""
        if self.running:
            logging.warning("Node already running")
            return
        
        self.running = True
        
        # Start worker threads
        self.tx_thread = threading.Thread(target=self._tx_worker, daemon=True)
        self.rx_thread = threading.Thread(target=self._rx_worker, daemon=True)
        
        self.tx_thread.start()
        self.rx_thread.start()
        
        logging.info(f"Node {self.node_id:016X} started")
    
    def stop(self):
        """Stop node operation"""
        if not self.running:
            return
        
        self.running = False
        
        # Wait for threads
        if self.tx_thread:
            self.tx_thread.join(timeout=2.0)
        if self.rx_thread:
            self.rx_thread.join(timeout=2.0)
        
        logging.info(f"Node {self.node_id:016X} stopped")
    
    def send_message(self, dest_id: int, message: str) -> bool:
        """
        Send message to destination
        
        Args:
            dest_id: Destination node ID
            message: Message text
            
        Returns:
            True if queued for transmission
        """
        return self.messaging.send_message(dest_id, message)
    
    def receive_message(self, timeout: Optional[float] = None):
        """
        Receive message
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (source_id, message) or None
        """
        return self.messaging.receive_message(timeout=timeout)
    
    def _tx_worker(self):
        """Transmission worker thread"""
        while self.running:
            try:
                packet = self.mesh.get_next_transmit(timeout=0.1)
                
                if packet is None:
                    continue
                
                # Serialize packet
                packet_bytes = packet.serialize()
                
                # Encrypt
                encrypted = self.security.encrypt_packet(packet_bytes)
                
                # Transmit via active modem
                with self.lock:
                    success = self.active_modem.transmit(encrypted)
                
                if success:
                    logging.debug(f"TX: {len(encrypted)} bytes to {packet.dest:016X}")
            
            except Exception as e:
                logging.error(f"TX error: {e}")
    
    def _rx_worker(self):
        """Reception worker thread"""
        while self.running:
            try:
                # Receive from active modem
                with self.lock:
                    data = self.active_modem.receive(timeout=1.0)
                
                if data is None:
                    # Send HELLO periodically
                    if self.discovery.should_send_hello():
                        hello = self.discovery.create_hello_packet()
                        self.mesh.queue_transmit(hello)
                    continue
                
                # Decrypt
                try:
                    packet_bytes = self.security.decrypt_packet(data)
                except ValueError:
                    logging.warning("Decryption failed, invalid packet")
                    continue
                
                # Parse packet
                try:
                    packet = HRCSPacket.deserialize(packet_bytes)
                except ValueError as e:
                    logging.warning(f"Deserialization failed: {e}")
                    continue
                
                # Update neighbor info (if HELLO)
                if packet.type == 'HELLO':
                    # Estimate RSSI and SNR (simplified)
                    rssi = -50  # Placeholder
                    snr = 30    # Placeholder
                    self.discovery.update_neighbor(packet.source, rssi, snr)
                    self.router.update_neighbor(packet.source, rssi, snr)
                
                # Process through mesh
                self.mesh.process_received_packet(packet)
                
                logging.debug(f"RX: From {packet.source:016X}")
            
            except Exception as e:
                logging.error(f"RX error: {e}")

