#!/usr/bin/env python3
"""
Main Orchestration Script for A-LMI System

Initializes all services, performs health checks, and coordinates
the autonomous agent loop with graceful shutdown handling.
"""

import sys
import time
import signal
import logging
import yaml
from pathlib import Path
from typing import Dict, Any
import threading

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from a_lmi.core.agent import ALMIAgent
from a_lmi.services.processing_core import ProcessingCore
from a_lmi.services.audio_processor import AudioProcessor
# Config loaded in ALMIOrchestrator


class ALMIOrchestrator:
    """
    Main orchestrator for the A-LMI system.
    
    Responsibilities:
    - Initialize all services
    - Health checks
    - Service coordination
    - Graceful shutdown
    - Error handling and recovery
    """
    
    def __init__(self, config_path: str = "infrastructure/config.yaml"):
        """
        Initialize orchestrator.
        
        Args:
            config_path: Path to configuration file
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('logs/a_lmi.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger.info("="*70)
        self.logger.info("A-LMI System - Unified Vibrational Intelligence")
        self.logger.info("="*70)
        
        # Service instances
        self.agent = None
        self.processing_core = None
        self.audio_processor = None
        
        # Threads
        self.service_threads = []
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Orchestrator initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()
        sys.exit(0)
    
    def check_infrastructure(self) -> bool:
        """
        Check infrastructure services are running.
        
        Returns:
            True if all services are up
        """
        self.logger.info("Checking infrastructure services...")
        
        checks = {
            'Kafka': self._check_kafka(),
            'MinIO': self._check_minio(),
            'Milvus': self._check_milvus(),
            'Neo4j': self._check_neo4j()
        }
        
        all_up = all(checks.values())
        
        for service, status in checks.items():
            status_str = "✓ UP" if status else "✗ DOWN"
            self.logger.info(f"  {service}: {status_str}")
        
        if not all_up:
            self.logger.error("Infrastructure services not ready. Please start with: docker-compose up -d")
            return False
        
        self.logger.info("All infrastructure services ready!")
        return True
    
    def _check_kafka(self) -> bool:
        """Check Kafka is running."""
        try:
            from kafka import KafkaProducer
            producer = KafkaProducer(
                bootstrap_servers=self.config['infrastructure']['kafka']['bootstrap_servers'],
                request_timeout_ms=5000
            )
            producer.close()
            return True
        except:
            return False
    
    def _check_minio(self) -> bool:
        """Check MinIO is running."""
        try:
            from minio import Minio
            client = Minio(
                self.config['infrastructure']['minio']['endpoint'],
                access_key=self.config['infrastructure']['minio']['access_key'],
                secret_key=self.config['infrastructure']['minio']['secret_key'],
                secure=False
            )
            client.list_buckets()
            return True
        except:
            return False
    
    def _check_milvus(self) -> bool:
        """Check Milvus is running."""
        try:
            from pymilvus import connections
            connections.connect(
                host=self.config['infrastructure']['milvus']['host'],
                port=self.config['infrastructure']['milvus']['port']
            )
            return True
        except:
            return False
    
    def _check_neo4j(self) -> bool:
        """Check Neo4j is running."""
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                self.config['infrastructure']['neo4j']['uri'],
                auth=(
                    self.config['infrastructure']['neo4j']['username'],
                    self.config['infrastructure']['neo4j']['password']
                )
            )
            with driver.session() as session:
                session.run("RETURN 1")
            driver.close()
            return True
        except:
            return False
    
    def initialize_services(self):
        """Initialize all A-LMI services."""
        self.logger.info("Initializing A-LMI services...")
        
        try:
            # Initialize agent
            self.logger.info("Initializing autonomous agent...")
            self.agent = ALMIAgent(self.config)
            
            # Initialize processing core
            self.logger.info("Initializing processing core...")
            self.processing_core = ProcessingCore()
            
            # Initialize audio processor (optional)
            try:
                self.logger.info("Initializing audio processor...")
                self.audio_processor = AudioProcessor(self.config)
            except Exception as e:
                self.logger.warning(f"Audio processor initialization failed: {e}")
                self.audio_processor = None
            
            self.logger.info("All services initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Error initializing services: {e}", exc_info=True)
            raise
    
    def start(self):
        """Start all services."""
        if self.running:
            self.logger.warning("Services already running")
            return
        
        self.logger.info("Starting A-LMI system...")
        
        # Check infrastructure first
        if not self.check_infrastructure():
            self.logger.error("Cannot start: infrastructure not ready")
            return
        
        # Initialize services
        try:
            self.initialize_services()
        except Exception as e:
            self.logger.error(f"Failed to initialize services: {e}")
            return
        
        # Start service threads
        self.running = True
        
        # Start agent (runs its own loops internally)
        if self.agent:
            agent_thread = threading.Thread(target=self.agent.run, daemon=True)
            agent_thread.start()
            self.service_threads.append(agent_thread)
        
        # Start processing core
        if self.processing_core:
            proc_thread = threading.Thread(target=self.processing_core.run, daemon=True)
            proc_thread.start()
            self.service_threads.append(proc_thread)
        
        # Start audio processor
        if self.audio_processor:
            audio_thread = threading.Thread(target=self._run_audio, daemon=True)
            audio_thread.start()
            self.service_threads.append(audio_thread)
        
        self.logger.info("="*70)
        self.logger.info("A-LMI System is RUNNING")
        self.logger.info("Press Ctrl+C to stop")
        self.logger.info("="*70)
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def _run_audio(self):
        """Run audio processor."""
        if self.audio_processor:
            self.audio_processor.start_recording()
            try:
                while self.running:
                    time.sleep(1)
            finally:
                self.audio_processor.stop_recording()
    
    def stop(self):
        """Stop all services gracefully."""
        if not self.running:
            return
        
        self.logger.info("Stopping A-LMI system...")
        self.running = False
        
        # Stop services
        if self.audio_processor:
            self.audio_processor.stop_recording()
        
        # Give threads time to finish
        for thread in self.service_threads:
            thread.join(timeout=5.0)
        
        self.logger.info("A-LMI system stopped")
    
    def status(self):
        """Print system status."""
        self.logger.info("="*70)
        self.logger.info("A-LMI System Status")
        self.logger.info("="*70)
        self.logger.info(f"Running: {self.running}")
        self.logger.info(f"Active threads: {len(self.service_threads)}")
        self.logger.info(f"Agent: {'✓' if self.agent else '✗'}")
        self.logger.info(f"Processing Core: {'✓' if self.processing_core else '✗'}")
        self.logger.info(f"Audio Processor: {'✓' if self.audio_processor else '✗'}")
        self.logger.info("="*70)


def main():
    """Main entry point."""
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Create orchestrator
    orchestrator = ALMIOrchestrator()
    
    # Start system
    orchestrator.start()


if __name__ == "__main__":
    main()

