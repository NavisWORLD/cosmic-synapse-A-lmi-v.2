"""
HRCS Protocol Packet Structure
Implements packet serialization, deserialization, and validation
"""

import struct
import hashlib
import time


class HRCSPacket:
    """
    HRCS protocol packet
    
    Packet Format:
    - Preamble (64 bits): Sync pattern taken
    - Header (256 bits): Version, type, hop count, sequence, source, dest, payload length, timestamp
    - Payload (0-4096 bytes): Application data
    - CRC-32 (32 bits): Error detection checksum
    """
    
    PACKET_TYPES = {
        'DATA': 0,
        'ACK': 1,
        'ROUTE': 2,
        'HELLO': 3
    }
    
    PACKET_TYPE_NAMES = {v: k for k, v in PACKET_TYPES.items()}
    
    def __init__(self, source, dest, payload, packet_type='DATA', version=1, sequence=None, timestamp=None):
        self.version = version
        self.type = packet_type
        self.hop_count = 0
        self.sequence = sequence or (int(time.time() * 1000000) & 0xFFFF)
        self.source = source
        self.dest = dest
        self.payload = payload if isinstance(payload, bytes) else payload.encode()
        self.timestamp = timestamp or int(time.time() * 1000000)
    
    def serialize(self):
        """
        Convert packet to bytes for transmission
        
        Returns:
            Byte string containing full packet
        """
        # Pack header in network byte order (big-endian)
        # Format: version(B), type(B), hop_count(H), sequence(H), source(Q), dest(Q), length(H), timestamp(Q)
        header = struct.pack('!BBHHQQHQ',
                           self.version,
                           self.type_to_int(),
                           self.hop_count,
                           self.sequence,
                           self.source,
                           self.dest,
                           len(self.payload),
                           self.timestamp)
        
        # Combine header and payload
        data = header + self.payload
        
        # Calculate CRC (using MD5 hash for simplicity, truncated to 4 bytes)
        crc = hashlib.md5(data).digest()[:4]
        
        return data + crc
    
    @staticmethod
    def deserialize(data):
        """
        Parse packet from bytes
        
        Args:
            data: Byte string containing packet
            
        Returns:
            HRCSPacket object
            
        Raises:
            ValueError: If CRC mismatch or malformed packet
        """
        if len(data) < 28 + 4:
            raise ValueError("Packet too short")
        
        # Verify CRC
        crc_received = data[-4:]
        data_without_crc = data[:-4]
        crc_calculated = hashlib.md5(data_without_crc).digest()[:4]
        
        if crc_received != crc_calculated:
            raise ValueError("CRC mismatch")
        
        # Parse header (32 bytes: B=1, B=1, H=2, H=2, Q=8, Q=8, H=2, Q=8)
        if len(data_without_crc) < 32:
            raise ValueError("Packet too short")
        header = struct.unpack('!BBHHQQHQ', data_without_crc[:32])
        payload = data_without_crc[32:]
        
        # Verify payload length matches header
        if len(payload) != header[6]:
            raise ValueError("Payload length mismatch")
        
        # Create packet object
        packet = HRCSPacket(header[4], header[5], payload)
        packet.version = header[0]
        packet.type = HRCSPacket.PACKET_TYPE_NAMES.get(header[1], 'DATA')
        packet.hop_count = header[2]
        packet.sequence = header[3]
        packet.timestamp = header[7]
        
        return packet
    
    def type_to_int(self):
        """Convert packet type string to integer"""
        return self.PACKET_TYPES.get(self.type, 0)
    
    def __repr__(self):
        return (f"HRCSPacket(type={self.type}, source={self.source:016X}, "
                f"dest={self.dest:016X}, hops={self.hop_count}, "
                f"payload_len={len(self.payload)})")

