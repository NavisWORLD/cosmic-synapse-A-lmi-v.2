#!/usr/bin/env python
"""
System Startup Script

Convenience script to start the entire A-LMI system.
"""

import sys
import subprocess
import time
import logging
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    required_modules = [
        'torch',
        'transformers',
        'sentence_transformers',
        'neo4j',
        'pymilvus',
        'dash',
        'plotly'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        logger.error(f"Missing dependencies: {missing}")
        logger.error("Run: pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies installed ✓")
    return True


def check_infrastructure():
    """Check if Docker infrastructure is running."""
    logger.info("Checking Docker infrastructure...")
    
    try:
        result = subprocess.run(
            ['docker', 'ps', '--format', '{{.Names}}'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        containers = result.stdout.strip().split('\n')
        required_containers = ['kafka', 'minio', 'milvus', 'neo4j']
        
        running = []
        missing = []
        
        for container in containers:
            for req in required_containers:
                if req in container.lower():
                    running.append(req)
        
        for req in required_containers:
            if req not in running:
                missing.append(req)
        
        if missing:
            logger.warning(f"Missing containers: {missing}")
            logger.info("Starting Docker Compose...")
            subprocess.run(['docker-compose', '-f', 'infrastructure/docker-compose.yml', 'up', '-d'])
            time.sleep(10)  # Wait for services to start
        else:
            logger.info("All containers running ✓")
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking Docker: {e}")
        logger.info("Make sure Docker is installed and running")
        return False


def initialize_databases():
    """Initialize database schemas."""
    logger.info("Initializing databases...")
    
    scripts = [
        'infrastructure/setup_kafka.py',
        'infrastructure/init_milvus.py',
        'infrastructure/init_neo4j.py'
    ]
    
    for script in scripts:
        if Path(script).exists():
            logger.info(f"Running {script}...")
            try:
                subprocess.run([sys.executable, script], check=True, timeout=60)
            except subprocess.TimeoutExpired:
                logger.warning(f"Script {script} timed out")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Script {script} failed: {e}")
        else:
            logger.warning(f"Script not found: {script}")


def start_monitoring(port: int = 8051):
    """Start monitoring dashboard."""
    logger.info("Starting monitoring dashboard...")
    
    try:
        from interfaces.monitoring import MetricsCollector, MonitoringDashboard
        
        collector = MetricsCollector()
        dashboard = MonitoringDashboard(collector, port=port)
        
        # Run in background
        import threading
        thread = threading.Thread(target=dashboard.run, args=(False, True))
        thread.daemon = True
        thread.start()
        
        logger.info(f"Monitoring dashboard running on http://localhost:{port}")
        return True
    except Exception as e:
        logger.warning(f"Could not start monitoring: {e}")
        return False


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("Unified Vibrational Intelligence System")
    print("="*70)
    print("\nStarting A-LMI system...\n")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check infrastructure
    if not check_infrastructure():
        sys.exit(1)
    
    # Initialize databases
    initialize_databases()
    
    # Start monitoring
    start_monitoring()
    
    print("\n✓ System infrastructure ready")
    print("\nStarting main A-LMI process...\n")
    print("="*70 + "\n")
    
    # Start main system
    try:
        from main import ALMIOrchestrator
        
        orchestrator = ALMIOrchestrator()
        orchestrator.run()
        
    except KeyboardInterrupt:
        print("\n\nShutting down system...")
    except Exception as e:
        logger.error(f"Error running system: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

