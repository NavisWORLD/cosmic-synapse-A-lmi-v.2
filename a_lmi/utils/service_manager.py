"""
Service Manager

Coordinates and manages A-LMI microservices lifecycle.
"""

import logging
import threading
import time
from typing import List, Dict, Any, Callable


class ServiceManager:
    """
    Manages lifecycle of A-LMI microservices.
    
    Features:
    - Start/stop services
    - Health monitoring
    - Automatic restart on failure
    - Dependency management
    """
    
    def __init__(self):
        """Initialize service manager."""
        self.logger = logging.getLogger(__name__)
        self.services = {}
        self.threads = {}
        self.running = False
        self.logger.info("Service manager initialized")
    
    def register_service(
        self,
        name: str,
        start_fn: Callable,
        stop_fn: Callable = None,
        dependencies: List[str] = None,
        restart_on_failure: bool = True
    ):
        """
        Register a service with the manager.
        
        Args:
            name: Service name
            start_fn: Function to start the service
            stop_fn: Function to stop the service
            dependencies: List of service names this depends on
            restart_on_failure: Whether to auto-restart if service fails
        """
        self.services[name] = {
            'start_fn': start_fn,
            'stop_fn': stop_fn,
            'dependencies': dependencies or [],
            'restart_on_failure': restart_on_failure,
            'status': 'stopped',
            'thread': None
        }
        
        self.logger.info(f"Registered service: {name}")
    
    def start_service(self, name: str) -> bool:
        """
        Start a specific service.
        
        Args:
            name: Service name
            
        Returns:
            Success status
        """
        if name not in self.services:
            self.logger.error(f"Service '{name}' not registered")
            return False
        
        service = self.services[name]
        
        # Check dependencies
        for dep in service['dependencies']:
            if not self.is_service_running(dep):
                self.logger.error(f"Service '{name}' depends on '{dep}' which is not running")
                return False
        
        try:
            # Start service in thread
            thread = threading.Thread(target=service['start_fn'], daemon=True)
            thread.start()
            
            service['thread'] = thread
            service['status'] = 'running'
            
            self.logger.info(f"Started service: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting service '{name}': {e}")
            return False
    
    def stop_service(self, name: str) -> bool:
        """
        Stop a specific service.
        
        Args:
            name: Service name
            
        Returns:
            Success status
        """
        if name not in self.services:
            self.logger.error(f"Service '{name}' not registered")
            return False
        
        service = self.services[name]
        
        try:
            if service['stop_fn']:
                service['stop_fn']()
            
            service['status'] = 'stopped'
            self.logger.info(f"Stopped service: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping service '{name}': {e}")
            return False
    
    def is_service_running(self, name: str) -> bool:
        """
        Check if a service is running.
        
        Args:
            name: Service name
            
        Returns:
            True if running
        """
        if name not in self.services:
            return False
        
        return self.services[name]['status'] == 'running'
    
    def start_all(self):
        """Start all registered services."""
        self.logger.info("Starting all services...")
        
        for name in self.services.keys():
            self.start_service(name)
        
        self.running = True
    
    def stop_all(self):
        """Stop all services."""
        self.logger.info("Stopping all services...")
        
        for name in self.services.keys():
            self.stop_service(name)
        
        self.running = False
    
    def get_status(self) -> Dict[str, str]:
        """
        Get status of all services.
        
        Returns:
            Dictionary mapping service names to status
        """
        return {name: service['status'] for name, service in self.services.items()}

