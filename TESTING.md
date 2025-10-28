# Testing and Validation Guide

## Setup Instructions

### 1. Start Infrastructure Services

```bash
cd infrastructure
docker-compose up -d
```

Wait for all services to be ready (especially Kafka, Neo4j, Milvus).

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment

Create a `.env` file:
```env
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
MINIO_ENDPOINT=localhost:9000
MILVUS_HOST=localhost
NEO4J_URI=bolt://localhost:7687
```

## Component Testing

### Test Light Token Creation

```bash
python -c "
from a_lmi.core.light_token import LightToken
import numpy as np

# Create test token
token = LightToken(
    source_uri='test://example',
    modality='text',
    raw_data_ref='ref://test',
    content_text='This is a test'
)

# Set embedding
embedding = np.random.rand(1536).astype(np.float32)
token.set_embedding(embedding)

print('Token created successfully!')
print(f'Spectral signature computed: {token._spectral_computed}')
print(f'Dominant frequency: {token.get_dominant_frequency()}')
"
```

### Test Vector Database

```bash
python -c "
from a_lmi.memory.vector_db_client import VectorDBClient
import yaml

with open('infrastructure/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

client = VectorDBClient(config)

# Test connection
print('Vector DB connected successfully!')
print(f'Collection: {client.collection_name}')
"
```

### Test Audio Processor

```bash
python a_lmi/services/audio_processor.py
```

Press Ctrl+C to stop.

### Test Web Crawler

```bash
python a_lmi/services/web_crawler.py
```

### Test Cosmic Synapse Simulation

```bash
python cosmic_synapse/cosmic_simulator.py
```

Press Ctrl+C to stop.

## Integration Testing

### Full Pipeline Test

1. Start all services
2. Run audio processor
3. Run web crawler with test URLs
4. Monitor Kafka topics
5. Verify Light Tokens are processed
6. Check storage in Milvus and Neo4j

## Validation Experiments

### Experiment 1: Spectral Signature Validation

**Hypothesis**: Spectral signatures cluster semantically related content.

**Procedure**:
1. Create Light Tokens for related content
2. Compute spectral similarities
3. Compare with semantic similarities
4. Analyze clustering

### Experiment 2: Cross-Modal Retrieval

**Hypothesis**: Spectral similarity enables cross-modal pattern discovery.

**Procedure**:
1. Create tokens for text, image, audio
2. Search for resonant frequencies
3. Verify Link between modalities

### Experiment 3: Stochastic Resonance

**Hypothesis**: Environmental audio affects simulation dynamics.

**Procedure**:
1. Run Cosmic Synapse with mic OFF
2. Run with mic ON
3. Compare particle clustering
4. Measure correlation with audio PSD

## Performance Benchmarking

### Vector Search Speed

```python
# Test vector search performance
import time
from a_lmi.memory.vector_db_client import VectorDBClient

client = VectorDBClient(config)

# Benchmark search
start = time.time()
results = client.search_semantic(query_embedding, limit=100)
end = time.time()

print(f'Search time: {end - start:.3f}s')
print(f'Results: {len(results)}')
```

### Token Processing Throughput

Measure tokens processed per second in the processing pipeline.

## Known Issues

1. Audio processor requires microphone access
2. Vosk model needs to be downloaded separately
3. Unity project requires manual setup

## Troubleshooting

### Kafka Connection Errors

```bash
# Check Kafka is running
docker ps | grep kafka

# Check Kafka logs
docker logs kafka
```

### Milvus Connection Errors

```bash
# Check Milvus is running
docker ps | grep milvus

# Restart Milvus
docker restart milvus-standalone
```

### Neo4j Connection Errors

```bash
# Check Neo4j is running
docker ps | grep neo4j

# Access Neo4j browser
# http://localhost:7474
# username: neo4j
# password: vibrational123
```

