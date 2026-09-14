from hrcs.core.packet import HRCSPacket
from hrcs.network.mesh import MeshNetwork


def test_mesh_forwards_new_packet_once_and_drops_replay():
    mesh = MeshNetwork(node_id=2)
    original = HRCSPacket(source=1, dest=3, payload=b"route", sequence=7, timestamp=123, version=2)
    mesh.process_received_packet(original)
    forwarded = mesh.get_next_transmit(timeout=0.01)
    assert forwarded is not None
    assert forwarded.hop_count == 1

    replay = HRCSPacket(source=1, dest=3, payload=b"route", sequence=7, timestamp=123, version=2)
    mesh.process_received_packet(replay)
    assert mesh.get_next_transmit(timeout=0.01) is None
