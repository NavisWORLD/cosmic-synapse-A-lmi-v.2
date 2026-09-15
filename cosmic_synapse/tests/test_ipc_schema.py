import pytest

from cosmic_synapse.ipc.schema import PROTOCOL_VERSION, decode_message, encode_message


def test_ipc_schema_round_trip_is_versioned():
    raw = encode_message("status", {"running": True})
    message = decode_message(raw)
    assert message["version"] == PROTOCOL_VERSION == 1
    assert message["type"] == "status"
    assert message["payload"] == {"running": True}


def test_ipc_schema_rejects_unknown_version():
    with pytest.raises(ValueError, match="version"):
        decode_message('{"version":999,"type":"status","payload":{}}')
