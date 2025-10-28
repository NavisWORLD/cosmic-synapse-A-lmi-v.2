"""
Network Simulator for HRCS
Simulates mesh network operation without hardware
"""

import time
import threading
from typing import Dict, List, Tuple
from dataclasses import dataclass

from hrcs.node import HRCSNode


@dataclass
class SimulatedNode:
    """Simulated HRCS node"""
    node: HRCSNode
    position: Tuple[float, float]  # x, y coordinates
    range: float  # Communication range
    connected: bool = True


class NetworkSimulator:
    """
    Simulates a mesh network of HRCS nodes
    
    Provides controlled environment for testing without physical hardware.
    """
    
    def __init__(self):
        """Initialize simulator"""
        self.nodes: Dict[int, SimulatedNode] = {}
        self.message_history: List[Tuple[int, int, str, float]] = []  # from, to, msg, time
        self.running = False
    
    def add_node(self, node_id: int, x: float, y: float, 
                 range: float = 100.0, **node_kwargs) -> SimulatedNode:
        """
        Add node to simulation
        
        Args:
            node_id: Node identifier
            x, y: Position coordinates
            range: Communication range in units
            **node_kwargs: Additional arguments for HRCSNode
            
        Returns:
            Simulated node
        """
        node = HRCSNode(node_id, acoustic_only=True, **node_kwargs)
        sim_node = SimulatedNode(node=node, position=(x, y), range=range)
        self.nodes[node_id] = sim_node
        return sim_node
    
    def get_distance(self, node_id1: int, node_id2: int) -> float:
        """Calculate distance between two nodes"""
        if node_id1 not in self.nodes or node_id2 not in self.nodes:
            return float('inf')
        
        n1 = self.nodes[node_id1]
        n2 = self.nodes[node_id2]
        
        dx = n1.position[0] - n2.position[0]
        dy = n1.position[1] - n2.position[1]
        
        return (dx**2 + dy**2)**0.5
    
    def are_in_range(self, node_id1: int, node_id2: int) -> bool:
        """Check if two nodes are in communication range"""
        distance = self.get_distance(node_id1, node_id2)
        
        # Both nodes must be within each other's range
        node1_range = self.nodes[node_id1].range
        node2_range = self.nodes[node_id2].range
        
        return distance <= node1_range and distance <= node2_range
    
    def send_message(self, from_id: int, to_id: int, message: str) -> bool:
        """
        Send message between nodes
        
        Args:
            from_id: Source node
            to_id: Destination node
            message: Message content
            
        Returns:
            True if message sent
        """
        if from_id not in self.nodes:
            return False
        
        self.nodes[from_id].node.send_message(to_id, message)
        
        # Record in history
        self.message_history.append((from_id, to_id, message, time.time()))
        
        return True
    
    def start_all(self):
        """Start all nodes"""
        for sim_node in self.nodes.values():
            sim_node.node.start()
        self.running = True
    
    def stop_all(self):
        """Stop all nodes"""
        for sim_node in self.nodes.values():
            sim_node.node.stop()
        self.running = False
    
    def get_network_topology(self) -> Dict[int, List[int]]:
        """Get current network topology (who can reach whom)"""
        topology = {}
        
        for node_id1 in self.nodes:
            topology[node_id1] = []
            for node_id2 in self.nodes:
                if node_id1 != node_id2 and self.are_in_range(node_id1, node_id2):
                    topology[node_id1].append(node_id2)
        
        return topology
    
    def print_topology(self):
        """Print network topology"""
        topology = self.get_network_topology()
        print("\nNetwork Topology:")
        print("=" * 40)
        for node_id, neighbors in topology.items():
            print(f"Node {node_id:016X}: {len(neighbors)} neighbors")
            for neighbor in neighbors:
                distance = self.get_distance(node_id, neighbor)
                print(f"  -> Node {neighbor:016X} (distance: {distance:.2f})")


def example_simulation():
    """Example network simulation"""
    print("HRCS Network Simulator")
    print("=" * 40)
    
    sim = NetworkSimulator()
    
    # Add some nodes in a line topology
    sim.add_node(0x0001, 0, 0, range=150)
    sim.add_node(0x0002, 100, 0, range=150)
    sim.add_node(0x0003, 200, 0, range=150)
    
    # Start all nodes
    sim.start_all()
    
    # Print topology
    sim.print_topology()
    
    # Test message
    sim.send_message(0x0001, 0x0003, "Hello from node 1!")
    
    # Run for a bit
    time.sleep(2)
    
    # Stop
    sim.stop_all()
    
    print(f"\nTotal messages sent: {len(sim.message_history)}")


if __name__ == "__main__":
    example_simulation()

