# HRCS Deployment Guide

## Quick Start

### 1. Installation

```bash
cd hrcs
pip install -r requirements.txt
pip install -e .
```

### 2. Configuration

Run setup wizard:
```bash
python scripts/setup_node.py
```

Or manually copy and edit:
```bash
cp config/config.yaml.example config/config.yaml
nano config/config.yaml
```

### 3. Start Node

```python
from hrcs.node import HRCSNode

node = HRCSNode(
    node_id=0x0000000000000001,
    network_key="your-secret-key",
    acoustic_only=True
)

node.start()

# Send message
node.send_message(0x0002, "Hello, World!")

# Receive
sender, message = node.receive_message(timeout=5.0)
print(f"From {sender}: {message}")
```

## Field Deployment

### Requirements

- Raspberry Pi 4 or equivalent
- Audio input/output (speaker, microphone)
- Optional: SDR hardware (LimeSDR, HackRF)
- Battery and solar power

### Steps

1. Install operating system
2. Install HRCS software
3. Configure node ID and network key
4. Deploy and test connectivity
5. Monitor network status

## Troubleshooting

- **No audio**: Check audio device permissions
- **No neighbors**: Verify other nodes are powered and in range
- **Messages not received**: Check network key matches
- **High latency**: Consider increasing routing update interval

