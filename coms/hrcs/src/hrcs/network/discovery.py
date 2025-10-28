"""
Neighbor Discovery Protocol
Automatic detection and management of nearby nodes
"""

import time
from typing import Dict, Tuple
from threading import Lock

from ..core.packet import HRCSPacket


class NeighborDiscovery:
    """
    Neighbor discovery and link quality monitoring
    
    Periodically sends HELLO packets and tracks neighbor availability.
    """
    
    HELLO_INTERVAL = 30  # seconds
    NEIGHBOR_TIMEOUT = 60  # seconds
    MAX_NEIGHBORS = 256
    
    def __init__(self, node_id: int):
        """
        Initialize neighbor discovery
        
        Args:
            node_id: This node's ID
        """
        self.node_id = node_id
        self.neighbors: Dict[int, Tuple[float, float, float]] = {}  # neighbor_id -> (rssi, snr, last_seen)
        self.lock = Lock()
        self.last_hello = 0
    
    def update_neighbor(self, neighbor_id: int, rssi: float, snr: float):
        """
        Update neighbor information
        
        Args:
            neighbor_id: Neighbor's node ID
            rssi: Received signal strength
            snr: Signal-to-noise ratio
        """
        with self.lock:
            self.neighbors[neighbor_id] = (rssi, snr, time.time())
    
    def should_send_hello(self) -> bool:
        """
        Check if HELLO packet should be sent
        
        Returns:
            True if should send
        """
        if time.time() - self.last_hello >= self.HELLO_INTERVAL:
            self.last_hello = time.time()
            return True
        return False
    
    def create_hello_packet(self) -> HRCSPacket:
        """
        Create HELLO packet for neighbor discovery
        
        Returns:
            HELLO packet (broadcast to all neighbors)
        """
        # Broadcast HELLO with empty payload
        return HRCSPacket(
            source=self.node_id,
            dest=0xFFFFFFFFFFFFFFFF,  # Broadcast address
            payload=b'',
            packet_type='HELLO'
        )
    
    def prune_dead_neighbors(self) -> Dict[int, Tuple[float, float, float]]:
        """
        Remove neighbors that haven't been heard from recently
        
        Returns:
            Dictionary of alive neighbors
        """
        current_time = time.time()
        dead_neighbors = []
        
        with self.lock:
            for neighbor_id, (rssi, snr, last_seen) in self.neighbors.items():
                if current_time - last_seen > self.NEIGHBOR_TIMEOUT:
                    dead_neighbors.append(neighbor_id)
            
            for neighbor_id in dead_neighbors:
                del self.neighbors[neighbor_id]
        
        return self.get_neighbors()
    
    def get_neighbors(self) -> Dict[int, Tuple[float, float, float]]:
        """
        Get current neighbor list
        
        Returns:
            Dictionary of neighbors
        """
        with self.lock:
            return dict(self.neighbors)
    
    def get_neighbor_count(self) -> int:
        """Get number of active neighbors"""
        with self.lock:
            return len(self.neighbors)

