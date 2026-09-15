# Harmonic Resonance Communication System (HRCS)

HRCS is a research communications prototype preserved inside the COSMIC SYNAPSE / A-LMI repository. It contains packet, authenticated-encryption, acoustic-modem, mesh-routing, simulated transport, and experimental SDR-radio code.

The restoration branch verifies deterministic software behavior. It does **not** claim proven infrastructure independence in all environments, golden-ratio performance superiority, anti-jamming superiority, field reliability, or production/emergency readiness.

## Current verified software surface

The root restoration CI exercises HRCS contracts for:

- versioned packet serialization and integrity checking;
- ChaCha20-Poly1305 authenticated symmetric encryption behavior;
- deterministic acoustic BPSK encode/decode round trips using software-generated samples;
- duplicate/replay handling and multi-hop forwarding logic;
- simulated end-to-end node communication;
- deterministic SHA-256-derived radio hop planning;
- transmit-side frequency retuning behavior through an injected/fake SDR boundary.

These are software results. They are not RF range, hardware robustness, regulatory, security-audit, or anti-jamming results.

## Installation

HRCS is a nested Python project and is not bundled into the root `cosmic-synapse-a-lmi` wheel.

```bash
cd coms/hrcs
python -m pip install -e .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest
```

The repository-level restoration workflow also runs selected HRCS tests with `PYTHONPATH=.:coms/hrcs/src`.

## Basic API

```python
from hrcs.node import HRCSNode

node = HRCSNode(node_id=0x0001, network_key="replace-with-a-test-key", acoustic_only=True)
node.start()

# A second compatible node/transport is required for real delivery.
node.send_message(0x0002, "Hello")

node.stop()
```

Use disposable test keys for examples. Do not treat a string embedded in source/config as a production key-management strategy.

## Architecture

```text
Application
  └─ messaging / CLI
Network
  └─ mesh / discovery / routing
Physical / transport
  ├─ acoustic modem
  ├─ simulated modem
  └─ experimental SDR radio modem
Core
  ├─ packet protocol
  ├─ authenticated symmetric crypto
  └─ mathematical/signal helpers
```

Historical names involving phi/golden-ratio, Lorenz dynamics, resonance, or vibrational information are retained as project lineage and software mechanisms. Their presence in an algorithm is not evidence that they provide a physical or communications advantage.

## Acoustic path

The restored acoustic contract checks modulation/demodulation in software-generated sample buffers. Live speaker/microphone operation depends on host audio devices, permissions, room acoustics, levels, noise, sample clocks, and platform libraries and therefore requires a separate hardware integration test.

## SDR radio path

The radio path:

- derives repeatable hop seeds from SHA-256 instead of Python's process-randomized `hash()`;
- produces deterministic channel plans;
- retunes the transmit frequency per planned hop when a compatible SDR boundary is available.

The restoration does **not** establish synchronized receive-side frequency hopping, anti-jamming performance, operating range, throughput, packet error rate, or compatibility with a particular SDR in the field.

Transmit only in a lawful configured test environment and comply with frequency, power, bandwidth, licensing, and equipment rules for your jurisdiction.

## Cryptography boundary

Authenticated symmetric encryption is implemented. A static/pre-shared key architecture does not provide cryptographic forward secrecy by itself.

The current tests are software contracts, not a third-party cryptographic audit.

## Status

**Research/alpha prototype.** Appropriate current uses are source review, deterministic simulation/software testing, and controlled laboratory integration work.

It should not be used as the sole communications path for safety-critical, emergency, commercial, or adversarial deployments without substantial independent engineering, hardware testing, protocol/security review, and regulatory work.

## License

HRCS contains its own historical `LICENSE` file titled `MIT License + Emergency Use Clause`. The 2026 restoration does not alter those nested legal terms. Because the repository root also contains a GPL-3.0 license, downstream redistribution of combined/derived work should receive appropriate legal review rather than relying on this README as licensing advice.

## Author

Cory Shane Davis

See the repository root `docs/CLAIMS_AND_LIMITATIONS.md`, `docs/SECURITY.md`, and `docs/REPRODUCIBILITY.md` for the restoration-wide evidence boundary.
