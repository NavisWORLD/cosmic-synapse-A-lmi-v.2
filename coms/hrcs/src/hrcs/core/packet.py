"""Versioned HRCS packet serialization and validation.

Packet v2 uses an explicit 8-byte synchronization/version preamble, a fixed
32-byte network-order header, a payload of at most 4096 bytes, and CRC-32 for
error detection. CRC-32 is not a cryptographic integrity mechanism; packet
authenticity is provided separately by ChaCha20-Poly1305.

Legacy v1 packets (no preamble + truncated MD5 checksum) remain readable and
serializable solely for historical compatibility.
"""

from __future__ import annotations

import hashlib
import struct
import time
import zlib


class HRCSPacket:
    PREAMBLE = b"HRCSv2\x00\x01"
    HEADER_FORMAT = "!BBHHQQHQ"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    CRC_SIZE = 4
    MAX_PAYLOAD = 4096

    PACKET_TYPES = {"DATA": 0, "ACK": 1, "ROUTE": 2, "HELLO": 3}
    PACKET_TYPE_NAMES = {value: key for key, value in PACKET_TYPES.items()}

    def __init__(
        self,
        source,
        dest,
        payload,
        packet_type="DATA",
        version=2,
        sequence=None,
        timestamp=None,
    ):
        payload_bytes = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
        if len(payload_bytes) > self.MAX_PAYLOAD:
            raise ValueError(f"HRCS payload exceeds {self.MAX_PAYLOAD} byte limit")
        if packet_type not in self.PACKET_TYPES:
            raise ValueError(f"Unknown HRCS packet type: {packet_type}")
        if version not in (1, 2):
            raise ValueError(f"Unsupported HRCS packet version: {version}")

        self.version = int(version)
        self.type = packet_type
        self.hop_count = 0
        self.sequence = (
            int(time.time() * 1_000_000) & 0xFFFF if sequence is None else int(sequence)
        )
        if not 0 <= self.sequence <= 0xFFFF:
            raise ValueError("HRCS sequence must fit in 16 bits")
        self.source = int(source)
        self.dest = int(dest)
        self.payload = payload_bytes
        self.timestamp = int(time.time() * 1_000_000) if timestamp is None else int(timestamp)

    def _header(self) -> bytes:
        return struct.pack(
            self.HEADER_FORMAT,
            self.version,
            self.type_to_int(),
            self.hop_count,
            self.sequence,
            self.source,
            self.dest,
            len(self.payload),
            self.timestamp,
        )

    def serialize(self) -> bytes:
        header_and_payload = self._header() + self.payload
        if self.version == 1:
            # Historical wire format retained for compatibility only.
            return header_and_payload + hashlib.md5(header_and_payload).digest()[:4]

        frame = self.PREAMBLE + header_and_payload
        crc = struct.pack("!I", zlib.crc32(frame) & 0xFFFFFFFF)
        return frame + crc

    @classmethod
    def deserialize(cls, data: bytes, *, allow_legacy: bool = True) -> "HRCSPacket":
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("Packet data must be bytes")
        raw = bytes(data)
        if raw.startswith(cls.PREAMBLE):
            return cls._deserialize_v2(raw)
        if allow_legacy:
            return cls._deserialize_v1(raw)
        raise ValueError("Missing HRCS v2 preamble")

    @classmethod
    def _deserialize_v2(cls, data: bytes) -> "HRCSPacket":
        minimum = len(cls.PREAMBLE) + cls.HEADER_SIZE + cls.CRC_SIZE
        if len(data) < minimum:
            raise ValueError("Packet too short")
        frame = data[:-cls.CRC_SIZE]
        received_crc = struct.unpack("!I", data[-cls.CRC_SIZE :])[0]
        calculated_crc = zlib.crc32(frame) & 0xFFFFFFFF
        if received_crc != calculated_crc:
            raise ValueError("CRC-32 mismatch")
        header_start = len(cls.PREAMBLE)
        header_end = header_start + cls.HEADER_SIZE
        header = struct.unpack(cls.HEADER_FORMAT, frame[header_start:header_end])
        if header[0] != 2:
            raise ValueError(f"Unsupported HRCS v2 frame version field: {header[0]}")
        return cls._from_header_and_payload(header, frame[header_end:])

    @classmethod
    def _deserialize_v1(cls, data: bytes) -> "HRCSPacket":
        if len(data) < cls.HEADER_SIZE + cls.CRC_SIZE:
            raise ValueError("Legacy packet too short")
        received = data[-cls.CRC_SIZE :]
        body = data[:-cls.CRC_SIZE]
        calculated = hashlib.md5(body).digest()[:4]
        if received != calculated:
            raise ValueError("Legacy checksum mismatch")
        header = struct.unpack(cls.HEADER_FORMAT, body[: cls.HEADER_SIZE])
        if header[0] != 1:
            raise ValueError("Missing HRCS v2 preamble")
        return cls._from_header_and_payload(header, body[cls.HEADER_SIZE :])

    @classmethod
    def _from_header_and_payload(cls, header, payload: bytes) -> "HRCSPacket":
        payload_length = header[6]
        if payload_length > cls.MAX_PAYLOAD:
            raise ValueError(f"HRCS payload exceeds {cls.MAX_PAYLOAD} byte limit")
        if len(payload) != payload_length:
            raise ValueError("Payload length mismatch")
        packet_type = cls.PACKET_TYPE_NAMES.get(header[1])
        if packet_type is None:
            raise ValueError(f"Unknown HRCS packet type id: {header[1]}")
        packet = cls(
            header[4],
            header[5],
            payload,
            packet_type=packet_type,
            version=header[0],
            sequence=header[3],
            timestamp=header[7],
        )
        packet.hop_count = header[2]
        return packet

    def type_to_int(self) -> int:
        return self.PACKET_TYPES[self.type]

    def __repr__(self) -> str:
        return (
            f"HRCSPacket(v={self.version}, type={self.type}, source={self.source:016X}, "
            f"dest={self.dest:016X}, hops={self.hop_count}, payload_len={len(self.payload)})"
        )
