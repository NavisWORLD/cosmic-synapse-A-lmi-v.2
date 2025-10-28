"""
Golden Ratio Distance Vector Routing
Modified DV routing using φ-optimization for path selection
"""

import time
from typing import Dict, Tuple, Optional

from ..core.math import PHI


class GoldenRatioRouter:
    """
    Routing using golden ratio path optimization
    
    Uses modified distance vector protocol where route quality is calculated
    using golden ratio metrics for optimal path selection with hysteresis.
    """
    
    def __init__(self, node_id: int):
        """
        Initialize router
        
        Args:
            node_id: This node's ID
        """
        self.node_id = node_id
        self.routing_table: Dict[int, Tuple[int, float]] = {}  # dest -> (next_hop, cost)
        self.neighbors: Dict[int, Tuple[float, float, float]] = {}  # neighbor_id -> (rssi, snr, last_seen)
    
    def calculate_path_cost(self, rssi: float, snr: float, hop_count: int) -> float:
        """
        Calculate path cost using φ-weighting
        
        Lower cost = better route
        
        Args:
            rssi: Received signal strength in dBm (-120 to 0)
            snr: Signal-to-noise ratio in dB (0 to 40)
            hop_count: Number of hops
            
        Returns:
            Path cost (lower is better)
        """
        # Normalize metrics (lower is better for cost)
        rssi_cost = (120 + rssi) / 120  # -120 to 0 dBm -> 0 to 1
        snr_cost = (40 - snr) / 40      # 0 to 40 dB -> 1 to 0
        hop_cost = hop_count / 10       # 0 to 10 hops -> 0 to 1
        
        # Golden ratio weighting
        # Emphasize good signal quality over hop count
        cost = (rssi_cost * PHI**2 + 
                snr_cost * PHI + 
                hop_cost * 1.0)
        
        return cost
    
    def update_neighbor(self, neighbor_id: int, rssi: float, snr: float):
        """
        Update neighbor information
        
        Args:
            neighbor_id: Neighbor's node ID
            rssi: Received signal strength
            snr: Signal-to-noise ratio
        """
        self.neighbors[neighbor_id] = (rssi, snr, time.time())
        
        # Direct neighbor has cost based on link quality
        cost = self.calculate_path_cost(rssi, snr, 1)
        self.routing_table[neighbor_id] = (neighbor_id, cost)
    
    def process_route_update(self, from_node: int, routes: Dict[int, Tuple[int, float]]):
        """
        Process routing update from neighbor
        
        Args:
            from_node: Node ID sending the update
            routes: Their routing table (dest -> (next_hop, cost))
        """
        if from_node not in self.neighbors:
            return
        
        neighbor_rssi, neighbor_snr, _ = self.neighbors[from_node]
        
        for dest, (next_hop, their_cost) in routes.items():
            if dest == self.node_id:
                continue  # Don't route to ourselves
            
            # Calculate cost through this neighbor
            link_cost = self.calculate_path_cost(neighbor_rssi, neighbor_snr, 1)
            total_cost = link_cost + their_cost
            
            # Update if better
            if dest not in self.routing_table:
                self.routing_table[dest] = (from_node, total_cost)
            else:
                current_cost = self.routing_table[dest][1]
                
                # Use φ for hysteresis (prevent route flapping)
                if total_cost < current_cost / PHI:
                    self.routing_table[dest] = (from_node, total_cost)
    
    def get_next_hop(self, dest: int) -> Optional[int]:
        """
        Get next hop for destination
        
        Args:
            dest: Destination node ID
            
        Returns:
            Next hop node ID, or None if no route
        """
        if dest in self.routing_table:
            return self.routing_table[dest][0]
        return None
    
    def prune_stale_routes(self, timeout: float = 30.0):
        """
        Remove routes through dead neighbors
        
        Args:
            timeout: Timeout in seconds for stale route removal
        """
        current_time = time.time()
        dead_neighbors = []
        
        for neighbor_id, (rssi, snr, last_seen) in self.neighbors.items():
            if current_time - last_seen > timeout:
                dead_neighbors.append(neighbor_id)
        
        # Remove dead neighbors
        for neighbor_id in dead_neighbors:
            del self.neighbors[neighbor_id]
            
            # Remove routes through dead neighbor
            for dest in list(self.routing_table.keys()):
                next_hop, cost = self.routing_table[dest]
                if next_hop == neighbor_id:
                    del self.routing_table[dest]
    
    def get_routing_table(self) -> Dict[int, Tuple[int, float]]:
        """Get current routing table copy"""
        return dict(self.routing_table)
    
    def get_neighbors(self) -> Dict[int, Tuple[float, float, float]]:
        """Get current neighbors copy"""
        return dict(self.neighbors)

