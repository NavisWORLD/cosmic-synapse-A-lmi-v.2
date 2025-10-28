"""
Tests for routing algorithms
"""

import pytest
from hrcs.network.routing import GoldenRatioRouter


class TestGoldenRatioRouter:
    """Test golden ratio routing"""
    
    def test_router_creation(self):
        """Test router creation"""
        router = GoldenRatioRouter(node_id=0x0001)
        assert router.node_id == 0x0001
    
    def test_calculate_path_cost(self):
        """Test path cost calculation"""
        router = GoldenRatioRouter(node_id=0x0001)
        
        # Good signal, close
        cost1 = router.calculate_path_cost(rssi=-50, snr=30, hop_count=1)
        
        # Poor signal, far
        cost2 = router.calculate_path_cost(rssi=-100, snr=10, hop_count=5)
        
        assert cost1 < cost2  # Lower cost is better
    
    def test_update_neighbor(self):
        """Test neighbor update"""
        router = GoldenRatioRouter(node_id=0x0001)
        router.update_neighbor(0x0002, rssi=-50, snr=30)
        
        neighbors = router.get_neighbors()
        assert 0x0002 in neighbors
    
    def test_get_next_hop(self):
        """Test getting next hop"""
        router = GoldenRatioRouter(node_id=0x0001)
        router.update_neighbor(0x0002, rssi=-50, snr=30)
        
        next_hop = router.get_next_hop(0x0002)
        assert next_hop == 0x0002
        
        # Unknown destination
        next_hop = router.get_next_hop(0x9999)
        assert next_hop is None
    
    def test_process_route_update(self):
        """Test processing route updates"""
        router = GoldenRatioRouter(node_id=0x0001)
        router.update_neighbor(0x0002, rssi=-50, snr=30)
        
        # Process update from neighbor 0x0002 about destination 0x0003
        routes = {
            0x0003: (0x0003, 1.0)
        }
        router.process_route_update(0x0002, routes)
        
        next_hop = router.get_next_hop(0x0003)
        assert next_hop == 0x0002
    
    def test_prune_stale_routes(self):
        """Test pruning stale routes"""
        router = GoldenRatioRouter(node_id=0x0001)
        router.update_neighbor(0x0002, rssi=-50, snr=30)
        
        # Prune with very short timeout (should remove all)
        router.prune_stale_routes(timeout=0.001)
        
        neighbors = router.get_neighbors()
        assert len(neighbors) == 0

