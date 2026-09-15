import asyncio

from cosmic_synapse.ipc.bridge import IPCBridge
from cosmic_synapse.ipc.schema import decode_message


class FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def send(self, message):
        self.sent.append(message)


def test_bridge_builds_versioned_spawn_command_without_running_event_loop():
    bridge = IPCBridge()
    raw = bridge.build_spawn_command(
        mass_type="star",
        position=(1.0, 2.0, 3.0),
        properties={"mass": 4.0},
        command_id="spawn-fixture",
    )
    decoded = decode_message(raw)
    assert decoded["version"] == 1
    assert decoded["type"] == "command"
    assert decoded["payload"]["command"] == "spawn_mass"
    assert decoded["payload"]["id"] == "spawn-fixture"
    assert decoded["payload"]["position"] == [1.0, 2.0, 3.0]


def test_process_raw_message_validates_schema_and_sends_versioned_ack():
    bridge = IPCBridge()
    websocket = FakeWebSocket()
    raw = bridge.build_spawn_command("star", (0, 0, 0), {}, command_id="abc")

    asyncio.run(bridge.process_raw_message(websocket, raw))

    assert len(websocket.sent) == 1
    ack = decode_message(websocket.sent[0])
    assert ack["type"] == "command_received"
    assert ack["payload"] == {"command_id": "abc"}
