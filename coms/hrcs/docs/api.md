# HRCS API Documentation

## Core Classes

### `HRCSNode`

Main node class for HRCS communication.

```python
from hrcs.node import HRCSNode

node = HRCSNode(node_id=0x0001, network_key="secret", acoustic_only=True)
```

**Methods:**
- `start()` - Start node operation
- `stop()` - Stop node
- `send_message(dest_id, message)` - Send message
- `receive_message(timeout)` - Receive message

### `AcousticModem`

Acoustic communication modem using OFDM.

```python
from hrcs.physical.acoustic import AcousticModem

modem = AcousticModem()
signal = modem.modulate(b"data")
data = modem.demodulate(signal)
```

### `RadioModem`

SDR radio modem with frequency hopping.

```python
from hrcs.physical.radio import RadioModem

modem = RadioModem(center_freq=433.92e6)
```

### `UnifiedMath`

Mathematical framework.

```python
from hrcs.core.math import UnifiedMath

# Generate golden ratio frequencies
freqs = UnifiedMath.golden_ratio_frequencies(432, 32)

# Generate Lorenz sequence
seq = UnifiedMath.lorenz_sequence(0, 0, 0, 1000)

# Stochastic enhancement
enhanced = UnifiedMath.stochastic_enhance(signal)

# Spectral encoding
spectrum = UnifiedMath.spectral_encode(b"data")
data = UnifiedMath.spectral_decode(spectrum)
```

### `HRCSPacket`

Packet structure.

```python
from hrcs.core.packet import HRCSPacket

packet = HRCSPacket(source=0x0001, dest=0x0002, payload=b"Hello")
data = packet.serialize()
packet2 = HRCSPacket.deserialize(data)
```

### `HRCSSecurity`

Encryption and security.

```python
from hrcs.core.crypto import HRCSSecurity

security = HRCSSecurity(pre_shared_key="password")
encrypted = security.encrypt_packet(b"data")
data = security.decrypt_packet(encrypted)
```

