"""
IPC Bridge between A-LMI and Cosmic Synapse

Enables bidirectional communication using WebSockets.
"""

import logging
import json
import asyncio
import time
from typing import Dict, Any, Optional, Callable
import websockets
from websockets.server import serve


class IPCBridge:
    """
    IPC bridge for A-LMI ↔ Cosmic Synapse communication.
    
    Protocol:
    - Commands from A-LMI to spawn masses
    - Status updates from simulation to A-LMI
    - Pattern data export
    - Synchronization
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        """
        Initialize IPC bridge.
        
        Args:
            host: Host to bind to
            port: Port for WebSocket server
        """
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = port
        self.clients = set()
        
        # Callbacks
        self.on_command_received: Optional[Callable[[Dict], None]] = None
        self.on_status_update: Optional[Callable[[Dict], None]] = None
        
        self.logger.info(f"IPC Bridge initialized on {host}:{port}")
    
    async def register_client(self, websocket):
        """Register a client connection."""
        self.clients.add(websocket)
        self.logger.info(f"Client connected. Total: {len(self.clients)}")
    
    async def unregister_client(self, websocket):
        """Unregister a client connection."""
        self.clients.discard(websocket)
        self.logger.info(f"Client disconnected. Total: {len(self.clients)}")
    
    async def handle_client(self, websocket, path):
        """Handle client connection."""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                data = json.loads(message)
                await self.process_message(websocket, data)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)
    
    async def process_message(self, websocket, data: Dict[str, Any]):
        """
        Process incoming message from client.
        
        Args:
            websocket: Client connection
            data: Message data
        """
        msg_type = data.get('type')
        
        if msg_type == 'command':
            # Command from A-LMI to simulation
            if self.on_command_received:
                self.on_command_received(data)
            
            # Echo back confirmation
            await websocket.send(json.dumps({
                'type': 'command_received',
                'command_id': data.get('id')
            }))
        
        elif msg_type == 'status':
            # Status update from simulation
            if self.on_status_update:
                self.on_status_update(data)
        
        elif msg_type == 'pattern_data':
            # Pattern data export from simulation
            self.logger.info(f"Received pattern data: {data.get('pattern_type')}")
        
        else:
            self.logger.warning(f"Unknown message type: {msg_type}")
    
    async def send_to_all(self, message: Dict[str, Any]):
        """
        Send message to all connected clients.
        
        Args:
            message: Message to send
        """
        if self.clients:
            data = json.dumps(message)
            await asyncio.gather(
                *[client.send(data) for client in self.clients],
                return_exceptions=True
            )
    
    def send_spawn_command(self, mass_type: str, position: tuple, properties: Dict[str, Any]):
        """
        Send spawn command to simulation.
        
        Args:
            mass_type: Type of mass ('star', 'black_hole')
            position: (x, y) position
            properties: Additional properties
        """
        command = {
            'type': 'command',
            'command': 'spawn_mass',
            'id': f"spawn_{int(time.time())}",
            'payload': {
                'mass_type': mass_type,
                'position': position,
                'properties': properties
            }
        }
        
        # Send asynchronously
        asyncio.create_task(self.send_to_all(command))
        self.logger.info(f"Sent spawn command: {mass_type} at {position}")
    
    async def run(self):
        """Run the IPC bridge server."""
        self.logger.info(f"Starting IPC bridge on ws://{self.host}:{self.port}")
        
        async with serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever


def main():
    """Test IPC bridge."""
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    bridge = IPCBridge()
    
    def on_command(cmd):
        print(f"Received command: {cmd}")
    
    bridge.on_command_received = on_command
    
    # Run server
    try:
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        print("\nStopping IPC bridge...")


if __name__ == "__main__":
    main()

