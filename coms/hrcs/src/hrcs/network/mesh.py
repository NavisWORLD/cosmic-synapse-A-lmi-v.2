"""
Mesh Networking Infrastructure
Multi-hop packet forwarding and network management
"""

import time
import queue
from typing import Dict, Set, Optional
from threading import Lock

from ..core.packet import HRCSPacket


class MeshNetwork:
    """
    Mesh networking for multi-hop packet delivery
    
    Handles packet forwarding, duplicate detection, and network topology management.
    """
    
    MAX_HOPS = 10
    PACKET_LIFETIME = 300  # seconds
    
    def __init__(self, node_id: int):
        """
        Initialize mesh network
        
        Args:
            node_id: This node's ID
        """
        self.node_id = node_id
        self.rx_queue = queue.Queue()
        self.tx_queue = queue.Queue()
        self.seen_packets: Dict[int, Set[int]] = {}  # node_id -> set of sequence numbers
        self.lock = Lock()
    
    def should_forward(self, packet: HRCSPacket) -> bool:
        """
        Determine if packet should be forwarded
        
        Args:
            packet: Packet to check
            
        Returns:
            True if should forward
        """
        # Don't forward if exceeded hop limit
        if packet.hop_count >= self.MAX_HOPS:
            return False
        
        # Don't forward if we've seen this packet before
        with self.lock:
            if packet.source in self.seen_packets:
                if packet.sequence in self.seen_packets[packet.source]:
                    return False
                self.seen_packets[packet.source].add(packet.sequence)
            else:
                self.seen_packets[packet.source] = {packet.sequence}
        
        # Don't forward if destined for us
        if packet.dest == self.node_id:
            return False
        
        return True
    
    def process_received_packet(self, packet: HRCSPacket):
        """
        Process a received packet
        
        Args:
            packet: Received packet
        """
        # Add to seen set
        with self.lock:
            if packet.source not in self.seen_packets:
                self.seen_packets[packet.source] = set()
            self.seen_packets[packet.source].add(packet.sequence)
        
        # Deliver if for us
        if packet.dest == self.node_id:
            self.rx_queue.put(packet)
        # Forward if should
        elif self.should_forward(packet):
            packet.hop_count += 1
            self.tx_queue.put(packet)
    
    def get_received_packet(self, timeout: Optional[float] = None) -> Optional[HRCSPacket]:
        """
        Get next received packet
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Received packet or None
        """
        try:
            return self.rx_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def queue_transmit(self, packet: HRCSPacket):
        """
        Queue packet for transmission
        
        Args:
            packet: Packet to queue
        """
        self.tx_queue.put(packet)
    
    def get_next_transmit(self, timeout: Optional[float] = None) -> Optional[HRCSPacket]:
        """
        Get next packet to transmit
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Packet to transmit or None
        """
        try:
            return self.tx_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def cleanup_old_entries(self):
        """Clean up old seen packet entries (periodic maintenance)"""
        with self.lock:
            # This is a simplified implementation
            # In production, would use time-stamped entries
            pass

