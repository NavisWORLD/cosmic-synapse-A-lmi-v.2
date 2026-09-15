from hrcs.node import HRCSNode
from hrcs.physical.simulated import SimulatedLink


def test_two_nodes_exchange_encrypted_message_over_simulated_transport():
    link = SimulatedLink()
    node1 = HRCSNode(1, network_key="fixture-key", modem=link.endpoint(1))
    node2 = HRCSNode(2, network_key="fixture-key", modem=link.endpoint(2))
    node1.start()
    node2.start()
    try:
        assert node1.send_message(2, "hello from cosmos")
        assert node2.receive_message(timeout=1.0) == (1, "hello from cosmos")
    finally:
        node1.stop()
        node2.stop()
