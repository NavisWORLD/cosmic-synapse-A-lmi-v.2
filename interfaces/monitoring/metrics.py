"""
Metrics Collection Service

Collects and aggregates system metrics for monitoring.
"""

import logging
import time
from typing import Dict, Any, List
from collections import defaultdict, deque
from datetime import datetime
import json


class MetricsCollector:
    """
    Collects system metrics for monitoring.
    
    Tracks:
    - Kafka message rates
    - Processing throughput
    - Memory usage
    - Error rates
    - Hypothesis generation stats
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.metrics = {
            'kafka': defaultdict(deque),
            'processing': defaultdict(deque),
            'memory': defaultdict(deque),
            'errors': defaultdict(int),
            'hypotheses': defaultdict(deque),
            'timestamps': deque()
        }
        
        # Configuration
        self.max_samples = 1000  # Keep last 1000 samples
        
        self.logger.info("Metrics collector initialized")
    
    def record_kafka_message(self, topic: str, size_bytes: int):
        """
        Record Kafka message metric.
        
        Args:
            topic: Topic name
            size_bytes: Message size in bytes
        """
        timestamp = time.time()
        
        self.metrics['kafka'][topic].append({
            'timestamp': timestamp,
            'size_bytes': size_bytes
        })
        
        # Limit deque size
        if len(self.metrics['kafka'][topic]) > self.max_samples:
            self.metrics['kafka'][topic].popleft()
    
    def record_processing(self, service: str, duration_ms: float, item_count: int = 1):
        """
        Record processing metric.
        
        Args:
            service: Service name
            duration_ms: Processing duration in milliseconds
            item_count: Number of items processed
        """
        timestamp = time.time()
        
        self.metrics['processing'][service].append({
            'timestamp': timestamp,
            'duration_ms': duration_ms,
            'item_count': item_count,
            'throughput': item_count / (duration_ms / 1000.0) if duration_ms > 0 else 0
        })
        
        if len(self.metrics['processing'][service]) > self.max_samples:
            self.metrics['processing'][service].popleft()
    
    def record_memory(self, layer: str, bytes_used: int, bytes_total: int):
        """
        Record memory usage.
        
        Args:
            layer: Memory layer name
            bytes_used: Bytes used
            bytes_total: Total bytes
        """
        timestamp = time.time()
        
        self.metrics['memory'][layer].append({
            'timestamp': timestamp,
            'bytes_used': bytes_used,
            'bytes_total': bytes_total,
            'percent_used': (bytes_used / bytes_total * 100) if bytes_total > 0 else 0
        })
        
        if len(self.metrics['memory'][layer]) > self.max_samples:
            self.metrics['memory'][layer].popleft()
    
    def record_error(self, service: str, error_type: str):
        """
        Record error.
        
        Args:
            service: Service name
            error_type: Type of error
        """
        key = f"{service}:{error_type}"
        self.metrics['errors'][key] += 1
    
    def record_hypothesis(self, hypothesis_id: str, score: float, category: str):
        """
        Record hypothesis generation.
        
        Args:
            hypothesis_id: Hypothesis identifier
            score: Confidence score
            category: Hypothesis category
        """
        timestamp = time.time()
        
        self.metrics['hypotheses'][category].append({
            'timestamp': timestamp,
            'hypothesis_id': hypothesis_id,
            'score': score
        })
        
        if len(self.metrics['hypotheses'][category]) > self.max_samples:
            self.metrics['hypotheses'][category].popleft()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary.
        
        Returns:
            Summary statistics
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'kafka': {},
            'processing': {},
            'memory': {},
            'errors': dict(self.metrics['errors']),
            'hypotheses': {}
        }
        
        # Kafka summary
        for topic, samples in self.metrics['kafka'].items():
            if samples:
                recent_samples = list(samples)[-60:]  # Last 60 samples
                summary['kafka'][topic] = {
                    'count': len(recent_samples),
                    'avg_size_bytes': sum(s['size_bytes'] for s in recent_samples) / len(recent_samples),
                    'total_bytes': sum(s['size_bytes'] for s in recent_samples)
                }
        
        # Processing summary
        for service, samples in self.metrics['processing'].items():
            if samples:
                recent_samples = list(samples)[-60:]
                summary['processing'][service] = {
                    'count': len(recent_samples),
                    'avg_duration_ms': sum(s['duration_ms'] for s in recent_samples) / len(recent_samples),
                    'avg_throughput': sum(s['throughput'] for s in recent_samples) / len(recent_samples),
                    'total_items': sum(s['item_count'] for s in recent_samples)
                }
        
        # Memory summary
        for layer, samples in self.metrics['memory'].items():
            if samples:
                latest = samples[-1]
                summary['memory'][layer] = {
                    'bytes_used': latest['bytes_used'],
                    'bytes_total': latest['bytes_total'],
                    'percent_used': latest['percent_used']
                }
        
        # Hypotheses summary
        for category, samples in self.metrics['hypotheses'].items():
            if samples:
                summary['hypotheses'][category] = {
                    'count': len(samples),
                    'avg_score': sum(s['score'] for s in samples) / len(samples)
                }
        
        return summary
    
    def export_to_file(self, filepath: str):
        """
        Export metrics to file.
        
        Args:
            filepath: Path to save metrics
        """
        summary = self.get_summary()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Metrics exported to {filepath}")
    
    def clear(self):
        """Clear all metrics."""
        self.metrics = {
            'kafka': defaultdict(deque),
            'processing': defaultdict(deque),
            'memory': defaultdict(deque),
            'errors': defaultdict(int),
            'hypotheses': defaultdict(deque),
            'timestamps': deque()
        }
        self.logger.info("Metrics cleared")


def main():
    """Test metrics collector."""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    collector = MetricsCollector()
    
    # Simulate some metrics
    for i in range(10):
        collector.record_kafka_message('light_tokens', 1024 * 10)
        collector.record_processing('processor', 100 + i, 1)
        collector.record_hypothesis(f'hyp_{i}', 0.7 + i * 0.03, 'pattern')
    
    # Get summary
    summary = collector.get_summary()
    
    print("\n" + "="*70)
    print("METRICS SUMMARY")
    print("="*70)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

