"""
Text Messaging Application
Send and receive text messages over the mesh network
"""

from typing import Optional, Tuple
import time

from ..core.packet import HRCSPacket


class Messaging:
    """
    Text messaging over HRCS network
    
    Provides simple send/receive interface for text messages.
    """
    
    def __init__(self, node_id: int, mesh_network):
        """
        Initialize messaging
        
        Args:
            node_id: This node's ID
            mesh_network: Mesh network instance for packet handling
        """
        self.node_id = node_id
        self.mesh = mesh_network
    
    def send_message(self, dest_id: int, message: str) -> bool:
        """
        Send text message to destination
        
        Args:
            dest_id: Destination node ID
            message: Message text
            
        Returns:
            True if queued for transmission
        """
        packet = HRCSPacket(
            source=self.node_id,
            dest=dest_id,
            payload=message.encode('utf-8'),
            packet_type='DATA'
        )
        
        self.mesh.queue_transmit(packet)
        return True
    
    def send_broadcast(self, message: str) -> bool:
        """
        Broadcast message to all nodes
        
        Args:
            message: Message text
            
        Returns:
            True if queued
        """
        packet = HRCSPacket(
            source=self.node_id,
            dest=0xFFFFFFFFFFFFFFFF,  # Broadcast address
            payload=message.encode('utf-8'),
            packet_type='DATA'
        )
        
        self.mesh.queue_transmit(packet)
        return True
    
    def receive_message(self, timeout: Optional[float] = None) -> Optional[Tuple[int, str]]:
        """
        Receive message
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (source_id, message) or None
        """
        packet = self.mesh.get_received_packet(timeout=timeout)
        
        if packet is None:
            return None
        
        try:
            message = packet.payload.decode('utf-8')
            return (packet.source, message)
        except UnicodeDecodeError:
            return None
    
    def format_message(self, source_id: int, message: str) -> str:
        """
        Format message for display
        
        Args:
            source_id: Source node ID
            message: Message text
            
        Returns:
            Formatted message string
        """
        return f"[{source_id:016X}] {message}"

