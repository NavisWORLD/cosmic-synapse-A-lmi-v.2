# Quick Start Guide

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- 16GB+ RAM recommended
- CUDA-compatible GPU (optional, for faster model inference)

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Additional Dependencies

```bash
# Install spaCy model for NER
python -m spacy download en_core_web_trf

# Download Vosk model (optional)
# wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
# unzip vosk-model-en-us-0.22.zip -d models/
```

### 3. Start Infrastructure Services

```bash
cd infrastructure
docker-compose up -d
```

Wait for all services to be ready (check with `docker ps`).

### 4. Initialize Databases

```bash
# Setup Kafka topics
python infrastructure/setup_kafka.py

# Initialize Milvus
python infrastructure/init_milvus.py

# Initialize Neo4j
python infrastructure/init_neo4j.py
```

### 5. Start the System

```bash
# Start A-LMI system
python main.py
```

## Usage

### 3D Knowledge Graph Visualization

In a separate terminal:

```bash
python -m interfaces.visualization.webapp
```

Open browser to: http://localhost:8050

### Cosmic Synapse Control Panel

In a separate terminal:

```bash
python -m interfaces.control_panel.app
```

Open browser to: http://localhost:8051

### Voice Interaction

The system will automatically capture audio if microphone is available. Speak naturally and the system will process your speech.

### Cosmic Synapse Simulation

Run the Python-based particle simulation:

```bash
python cosmic_synapse/cosmic_simulator.py
```

## Configuration

Edit `infrastructure/config.yaml` to customize:
- Service endpoints
- Model parameters
- Security settings
- Audio processing parameters

## Troubleshooting

### Kafka Connection Issues

```bash
docker logs kafka
# Ensure Kafka is running on port 9092
```

### Milvus Connection Issues

```bash
docker logs milvus-standalone
# Ensure Milvus is running on port 19530
```

### Neo4j Connection Issues

```bash
docker logs neo4j
# Access Neo4j browser at http://localhost:7474
# username: neo4j
# password: vibrational123
```

### Audio Processor Issues

Make sure microphone permissions are granted and Vosk model is downloaded.

## Next Steps

- Read the full documentation in `README.md`
- Check `IMPLEMENTATION_SUMMARY.md` for system overview
- Review `TESTING.md` for validation experiments
- See `FINAL_STATUS.md` for what's implemented

