"""
Experiment 2: Frequency-Dependent Recall

Test whether environmental acoustic conditions affect AI performance
through stochastic resonance mechanisms.

Hypothesis: Recall accuracy improves under frequency-matched conditions,
demonstrating state-dependent memory mediated by acoustic frequency.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from scipy import stats
import json
from pathlib import Path


class FrequencyRecallExperiment:
    """
    Experiment to test frequency-dependent recall effects.
    """
    
    def __init__(self, output_dir: str = "experiments/frequency_recall/results/"):
        """Initialize experiment."""
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_test_data(self, n_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Create test knowledge to ingest.
        
        Args:
            n_samples: Number of test samples
            
        Returns:
            List of test documents
        """
        samples = []
        
        for i in range(n_samples):
            samples.append({
                'id': f"test_{i}",
                'text': f"Test document {i} with unique information.",
                'category': f"category_{i % 5}"
            })
        
        return samples
    
    def simulate_ingestion_conditions(self, test_data: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Simulate ingesting data under different acoustic conditions.
        
        Args:
            test_data: Test documents
            
        Returns:
            Dictionary mapping conditions to ingested data
        """
        conditions = {
            'quiet': [],
            'white_noise': [],
            'tone_432hz': [],
            'tone_1000hz': []
        }
        
        # In production, would actual ingest with audio context
        for doc in test_data:
            for condition in conditions.keys():
                ingested = doc.copy()
                ingested['acoustic_context'] = condition
                ingested['frequency_band'] = self._get_frequency_band(condition)
                conditions[condition].append(ingested)
        
        return conditions
    
    def _get_frequency_band(self, condition: str) -> str:
        """Get dominant frequency band for condition."""
        bands = {
            'quiet': 'low',
            'white_noise': 'broad',
            'tone_432hz': '440hz_band',
            'tone_1000hz': '1khz_band'
        }
        return bands.get(condition, 'unknown')
    
    def test_recall(
        self,
        ingested_data: Dict[str, List[Dict]],
        query: str,
        matched_condition: str
    ) -> Dict[str, float]:
        """
        Test recall accuracy under matched vs mismatched conditions.
        
        Args:
            ingested_data: Data ingested under different conditions
            query: Query string
            matched_condition: Condition to test under
            
        Returns:
            Recall scores for different conditions
        """
        scores = {}
        
        # Test under matched condition
        matched_data = ingested_data[matched_condition]
        matched_score = self._calculate_recall(matched_data, query)
        scores['matched'] = matched_score
        
        # Test under mismatched conditions
        for condition in ingested_data.keys():
            if condition != matched_condition:
                mismatched_data = ingested_data[condition]
                mismatched_score = self._calculate_recall(mismatched_data, query)
                scores[f'mismatched_{condition}'] = mismatched_score
        
        # Calculate average mismatched
        mismatched_scores = [s for k, s in scores.items() if k.startswith('mismatched_')]
        scores['mismatched_avg'] = np.mean(mismatched_scores) if mismatched_scores else 0
        
        return scores
    
    def _calculate_recall(self, data: List[Dict], query: str) -> float:
        """
        Calculate recall score for data given query.
        
        Args:
            data: Test data
            query: Query string
            
        Returns:
            Recall score
        """
        # Simplified recall calculation
        # In production, would use actual vector search
        
        matches = sum(1 for doc in data if query.lower() in doc['text'].lower())
        recall = matches / len(data) if data else 0
        
        return recall
    
    def run(self) -> Dict[str, Any]:
        """Run complete experiment."""
        self.logger.info("Starting frequency-dependent recall experiment...")
        
        # Create test data
        test_data = self.create_test_data()
        
        # Simulate ingestion under different conditions
        ingested = self.simulate_ingestion_conditions(test_data)
        
        # Test recall
        query = "test document"
        results = {}
        
        for condition in ['quiet', 'white_noise', 'tone_432hz', 'tone_1000hz']:
            recall_scores = self.test_recall(ingested, query, condition)
            results[condition] = recall_scores
        
        # Statistical analysis
        all_matched = [results[c]['matched'] for c in results.keys()]
        all_mismatched = [results[c]['mismatched_avg'] for c in results.keys()]
        
        # Perform t-test
        t_stat, p_value = stats.ttest_rel(all_matched, all_mismatched)
        
        experiment_results = {
            'experiment': 'Frequency-Dependent Recall',
            'hypothesis': 'Frequency-matched recall outperforms mismatched',
            'n_samples': len(test_data),
            'conditions_tested': list(results.keys()),
            'results': results,
            'statistics': {
                'matched_mean': float(np.mean(all_matched)),
                'mismatched_mean': float(np.mean(all_mismatched)),
                'difference': float(np.mean(all_matched) - np.mean(all_mismatched)),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            },
            'conclusion': 'Significant difference' if p_value < 0.05 else 'No significant difference'
        }
        
        # Save results
        results_file = self.output_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        self.logger.info(f"Experiment complete. Results saved to {results_file}")
        
        return experiment_results


def main():
    """Run the experiment."""
    logging.basicConfig(level=logging.INFO)
    
    experiment = FrequencyRecallExperiment()
    results = experiment.run()
    
    print("\n" + "="*70)
    print("FREQUENCY-DEPENDENT RECALL EXPERIMENT RESULTS")
    print("="*70)
    print(f"Matched recall: {results['statistics']['matched_mean']:.3f}")
    print(f"Mismatched recall: {results['statistics']['mismatched_mean']:.3f}")
    print(f"Difference: {results['statistics']['difference']:.3f}")
    print(f"p-value: {results['statistics']['p_value']:.4f}")
    print(f"Significant: {results['statistics']['significant']}")
    print(f"Conclusion: {results['conclusion']}")


if __name__ == "__main__":
    main()

