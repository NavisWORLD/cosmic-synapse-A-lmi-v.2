"""
Base Modem Interface
Abstract base class for all modem implementations
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseModem(ABC):
    """
    Abstract base class for all HRCS modem implementations
    
    All modems must implement these methods to interface with the network layer.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize modem
        
        Args:
            enabled: Whether modem is currently enabled
        """
        self.enabled = enabled
        self.sample_rate: Optional[int] = None
        self.band_name: str = "unknown"
    
    @abstractmethod
    def transmit(self, data: bytes) -> bool:
        """
        Transmit data bytes
        
        Args:
            data: Data to transmit
            
        Returns:
            True if transmission succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    def receive(self, timeout: float = 1.0) -> Optional[bytes]:
        """
        Receive data bytes
        
        Args:
            timeout: Maximum time to wait for data in seconds
            
        Returns:
            Received data bytes, or None if timeout
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if modem hardware is available
        
        Returns:
            True if hardware present and functional
        """
        pass
    
    def get_info(self) -> dict:
        """
        Get modem information
        
        Returns:
            Dictionary with modem information
        """
        return {
            'type': self.__class__.__name__,
            'enabled': self.enabled,
            'available': self.is_available(),
            'band': self.band_name,
            'sample_rate': self.sample_rate
        }

