"""
Command Line Interface for HRCS
Interactive CLI for node operation and testing
"""

import sys
from typing import Optional

try:
    import click
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False


def print_status(message: str):
    """Print status message"""
    print(f"[HRCS] {message}")


def print_error(message: str):
    """Print error message"""
    print(f"[ERROR] {message}", file=sys.stderr)


if CLICK_AVAILABLE:
    @click.group()
    def cli():
        """HRCS Command Line Interface"""
        pass
    
    @cli.command()
    def status():
        """Show node status"""
        click.echo("HRCS Node Status")
        click.echo("=" * 40)
        click.echo("Node ID: Not initialized")
        click.echo("Status: Offline")
    
    @cli.command()
    @click.argument('dest', type=str)
    @click.argument('message')
    def send(dest, message):
        """Send message to destination"""
        click.echo(f"Sending message to {dest}: {message}")
    
    @cli.command()
    def recv():
        """Receive messages"""
        click.echo("Listening for messages...")
else:
    def cli():
        """Fallback CLI when click not available"""
        print("Click library not installed. Install with: pip install click")

