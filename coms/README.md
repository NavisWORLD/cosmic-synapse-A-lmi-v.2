# Harmonic Resonance Communication System (HRCS)

**Infrastructure-Free Communication Through Vibrational Information Dynamics**

[![License](https://img.shields.io/badge/License-MIT%20%2B%20Emergency-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Alpha-orange.svg)](https://github.com/your-repo/hrcs)

## Overview

HRCS is a complete, buildable communication system requiring **ZERO infrastructure** - no towers, no satellites, no internet, no power grid dependency. It operates purely through **frequency-domain information encoding** based on the Unified Vibrational Information Theory.

### Key Features

- 🌊 **Multi-Band Operation**: Acoustic (20Hz-20kHz), VHF (30-300 MHz), UHF (300-3000 MHz)
- 📡 **Mesh Networking**: Automatic peer-to-peer mesh formation with multi-hop routing
- 🔐 **Strong Encryption**: ChaCha20-Poly1305 authenticated encryption
- 📈 **Golden Ratio Optimization**: φ-based frequency spacing for optimal performance
- 🌪️ **Lorenz Frequency Hopping**: Chaotic anti-jamming spread spectrum
- 🔊 **Stochastic Resonance**: Signal enhancement in noisy environments
- ⚡ **Acoustic-First**: Works fully without any SDR hardware (uses audio only)

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/hrcs.git
cd hrcs

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Basic Usage

```python
from hrcs.node import HRCSNode

# Create and start node
node = HRCSNode(node_id=0x0001, network_key="secret-key", acoustic_only=True)
node.start()

# Send message
node.send_message(0x0002, "Hello, world!")

# Receive message
sender, message = node.receive_message(timeout=5.0)
print(f"From {sender}: {message}")

# Stop node
node.stop()
```

### CLI Usage

```bash
# Show status
hrcs status

# Send message
hrcs send 0x0002 "Test message"

# Receive messages
hrcs recv
```

## System Requirements

### Minimum (Acoustic Only)
- Python 3.8+
- Audio input/output (speaker, microphone)
- Standard PC/Laptop

### Recommended (Full Features)
- Raspberry Pi 4 or equivalent
- LimeSDR Mini or HackRF One (SDR)
- High-quality audio interface
- Solar power system for field deployment

## Architecture

```
┌─────────────────────────────────────┐
│  Application Layer                  │
│  - Messaging, CLI                   │
├─────────────────────────────────────┤
│  Network Layer                      │
│  - Mesh Networking                  │
│  - Golden Ratio Routing             │
│  - Neighbor Discovery               │
├─────────────────────────────────────┤
│  Physical Layer                     │
│  - Acoustic OFDM Modem             │
│  - SDR Radio Modem                 │
│  - Spectral Encoding                │
├─────────────────────────────────────┤
│  Core                               │
│  - Math Framework (φ, Lorenz, SR)  │
│  - Packet Protocol                  │
│  - Security (ChaCha20-Poly1305)    │
└─────────────────────────────────────┘
```

## Use Cases

- 🚨 **Emergency Communications**: When infrastructure fails
- 🏔️ **Remote Areas**: No cellular coverage
- 🛡️ **Resilience**: EMP/solar storm preparation
- 🎯 **Secure Mesh**: Private networks without internet
- 🔬 **Research**: Frequency-domain communication experiments

## Documentation

- [Theory](docs/theory.md) - Mathematical foundation
- [Hardware Guide](docs/hardware.md) - Build your own node
- [Deployment](docs/deployment.md) - Field deployment guide
- [API Reference](docs/api.md) - Complete API documentation

## Development

### Running Tests

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# With coverage
pytest --cov=hrcs
```

### Code Quality

```bash
# Format code
black src/

# Lint
flake8 src/
pylint src/

# Type checking
mypy src/
```

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Emergency Use

**As stated in the license**, in declared emergencies or infrastructure failures, all restrictions are waived for humanitarian deployment.

## License

MIT License + Emergency Use Clause - See [LICENSE](LICENSE) for details.

## Credits

**Author**: Cory Shane Davis  
**Based on**: The Unified Theory of Vibrational Information Processing  
**Publication Date**: October 28, 2025

## Status

⚠️ **Alpha Stage**: Core functionality implemented, field testing in progress.  
📋 **Ready for**: Laboratory testing, prototype deployment, research use.  
🚧 **Not Ready for**: Production critical systems, commercial deployment.

## Disclaimer

This system is provided for research, education, and emergency preparedness purposes. Users must comply with all applicable regulations regarding radio frequency transmission in their jurisdiction.

---

*"In the frequency, we find freedom."*

