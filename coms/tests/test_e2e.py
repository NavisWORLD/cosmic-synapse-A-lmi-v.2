"""
End-to-end tests for HRCS
"""

import time
import pytest
from hrcs.node import HRCSNode


class TestTwoNodeCommunication:
    """Test basic two-node communication"""
    
    def test_node_creation(self):
        """Test creating two nodes"""
        node1 = HRCSNode(0x0001, acoustic_only=True)
        node2 = HRCSNode(0x0002, acoustic_only=True)
        
        assert node1.node_id == 0x0001
        assert node2.node_id == 0x0002
        
        # Cleanup
        node1.stop()
        node2.stop()
    
    def test_node_start_stop(self):
        """Test starting and stopping nodes"""
        node = HRCSNode(0x0001, acoustic_only=True)
        
        node.start()
        assert node.running == True
        
        time.sleep(0.1)
        
        node.stop()
        assert node.running == False

# Note: Full end-to-end tests requiring actual audio hardware
# should be run manually in a controlled environment

