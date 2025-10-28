"""
Tests for packet structure
"""

import pytest
from hrcs.core.packet import HRCSPacket


class TestPacketSerialization:
    """Test packet serialization and deserialization"""
    
    def test_serialize(self):
        """Test packet serialization"""
        packet = HRCSPacket(
            source=0x0001,
            dest=0x0002,
            payload=b"test",
            packet_type='DATA'
        )
        data = packet.serialize()
        assert isinstance(data, bytes)
        assert len(data) > 0
    
    def test_round_trip(self):
        """Test serialize/deserialize round trip"""
        packet1 = HRCSPacket(
            source=0x0001,
            dest=0x0002,
            payload=b"Hello, World!",
            packet_type='DATA'
        )
        data = packet1.serialize()
        packet2 = HRCSPacket.deserialize(data)
        
        assert packet2.source == packet1.source
        assert packet2.dest == packet1.dest
        assert packet2.payload == packet1.payload
        assert packet2.type == packet1.type
    
    def test_crc_verification(self):
        """Test CRC verification"""
        packet = HRCSPacket(
            source=0x0001,
            dest=0x0002,
            payload=b"test",
            packet_type='DATA'
        )
        data = packet.serialize()
        
        # Corrupt data
        corrupted = data[:-1] + b'\x00'
        
        with pytest.raises(ValueError):
            HRCSPacket.deserialize(corrupted)
    
    def test_different_types(self):
        """Test different packet types"""
        types = ['DATA', 'ACK', 'ROUTE', 'HELLO']
        for ptype in types:
            packet = HRCSPacket(
                source=0x0001,
                dest=0x0002,
                payload=b"",
                packet_type=ptype
            )
            data = packet.serialize()
            assert len(data) > 0

