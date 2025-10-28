# HRCS Hardware Guide

## Bill of Materials

### Minimum (Acoustic Only)
- Raspberry Pi 4 (4GB RAM) - $55
- USB Audio Interface (24-bit/96kHz) - $25
- Speaker (10W, 20Hz-20kHz) - $30
- Microphone (omnidirectional) - $40
- Battery (LiFePO4 12V 100Ah) - $900
- Solar Panel (100W) - $120
- **Total: ~$1,170**

### Complete (With SDR)
- Above plus:
- LimeSDR Mini - $159
- Antennas (VHF/UHF) - $95
- **Total: ~$1,424**

## Assembly

See `../HRCS_Hardware_Assembly_Guide.md` for detailed assembly instructions.

## Software Installation

```bash
# On Raspberry Pi
sudo apt update
sudo apt install python3-pip python3-numpy
pip3 install -r requirements.txt
```

## Testing

```bash
# Test acoustic modem
python -c "from hrcs.physical.acoustic import AcousticModem; m=AcousticModem(); print(m.is_available())"

# Test complete node
python -c "from hrcs.node import HRCSNode; n=HRCSNode(0x0001, acoustic_only=True); print('OK')"
```

