import pytest

from hrcs.core.packet import HRCSPacket


def test_packet_v2_round_trip_has_preamble_and_crc32():
    packet = HRCSPacket(1, 2, b"hello", sequence=0, timestamp=123, version=2)
    encoded = packet.serialize()
    assert encoded.startswith(HRCSPacket.PREAMBLE)
    restored = HRCSPacket.deserialize(encoded)
    assert restored.source == 1
    assert restored.dest == 2
    assert restored.payload == b"hello"
    assert restored.sequence == 0
    assert restored.timestamp == 123


def test_packet_v2_rejects_corruption():
    packet = HRCSPacket(1, 2, b"hello", version=2)
    encoded = bytearray(packet.serialize())
    encoded[-5] ^= 0x01
    with pytest.raises(ValueError, match="CRC"):
        HRCSPacket.deserialize(bytes(encoded))


def test_packet_rejects_payload_over_protocol_limit():
    with pytest.raises(ValueError, match="4096"):
        HRCSPacket(1, 2, b"x" * 4097, version=2)
