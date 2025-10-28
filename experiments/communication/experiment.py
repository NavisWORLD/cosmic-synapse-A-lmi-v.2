"""
Experiment 4: Communication Frequency Matching

Test whether frequency-matched delivery improves user comprehension
independent of semantic content.

Hypothesis: Delivery with specific prosodic patterns produces better
outcomes independent of semantic content.
"""

import logging
import numpy as np
from typing import Dict, Any, List
import json
from pathlib import Path


class CommunicationFrequencyExperiment:
    """
    Experiment to test communication frequency matching effects.
    """
    
    def __init__(self, output_dir: str = "experiments/communication/results/"):
        """Initialize experiment."""
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_speech_variants(self, text: str, n_variants: int = 10) -> List[Dict[str, Any]]:
        """
        Generate speech variants with different prosodic features.
        
        Args:
            text: Text content
            n_variants: Number of variants to generate
            
        Returns:
            List of speech variants with features
        """
        variants = []
        
        prosodic_features = [
            {'pitch': 120, 'tempo': 150, 'rhythm': 'regular'},
            {'pitch': 140, 'tempo': 160, 'rhythm': 'irregular'},
            {'pitch': 100, 'tempo': 130, 'rhythm': 'regular'},
            {'pitch': 115, 'tempo': 145, 'rhythm': 'irregular'},
            {'pitch': 135, 'tempo': 155, 'rhythm': 'variable'},
        ]
        
        for i in range(n_variants):
            features = prosodic_features[i % len(prosodic_features)]
            variants.append({
                'id': f"variant_{i}",
                'text': text,
                'pitch': features['pitch'],
                'tempo': features['tempo'],
                'rhythm': features['rhythm']
            })
        
        return variants
    
    def simulate_comprehension_test(self, variants: List[Dict]) -> List[Dict]:
        """
        Simulate comprehension testing on variants.
        
        Args:
            variants: Speech variants to test
            
        Returns:
            Variants with comprehension scores
        """
        results = []
        
        for variant in variants:
            # Simulate comprehension score based on prosodic features
            # Higher pitch and regular rhythm tend to score better (hypothesis)
            pitch_score = variant['pitch'] / 200  # Normalize
            tempo_score = 0.5  # Neutral
            rhythm_score = 0.8 if variant['rhythm'] == 'regular' else 0.5
            
            # Combined score
            comprehension = (pitch_score + tempo_score + rhythm_score) / 3
            comprehension += np.random.uniform(-0.1, 0.1)  # Add noise
            
            result = variant.copy()
            result['comprehension_score'] = comprehension
            result['retention_score'] = comprehension * 0.9  # Retention slightly lower
            results.append(result)
        
        return results
    
    def identify_optimal_pattern(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Identify optimal prosodic pattern.
        
        Args:
            results: Test results
            
        Returns:
            Optimal pattern characteristics
        """
        # Find variant with highest comprehension
        best = max(results, key=lambda x: x['comprehension_score'])
        
        optimal_pattern = {
            'pitch': best['pitch'],
            'tempo': best['tempo'],
            'rhythm': best['rhythm'],
            'comprehension_score': best['comprehension_score'],
            'retention_score': best['retention_score']
        }
        
        return optimal_pattern
    
    def run(self) -> Dict[str, Any]:
        """Run complete experiment."""
        self.logger.info("Starting communication frequency matching experiment...")
        
        # Generate variants
        test_text = "The golden ratio appears in many natural phenomena."
        variants = self.generate_speech_variants(test_text, n_variants=20)
        
        # Test comprehension
        results = self.simulate_comprehension_test(variants)
        
        # Identify optimal pattern
        optimal = self.identify_optimal_pattern(results)
        
        # Calculate statistics
        comprehension_scores = [r['comprehension_score'] for r in results]
        
        experiment_results = {
            'experiment': 'Communication Frequency Matching',
            'hypothesis': 'Prosodic features affect comprehension independently',
            'n_variants': len(variants),
            'optimal_pattern': optimal,
            'statistics': {
                'mean_comprehension': float(np.mean(comprehension_scores)),
                'std_comprehension': float(np.std(comprehension_scores)),
                'min_comprehension': float(np.min(comprehension_scores)),
                'max_comprehension': float(np.max(comprehension_scores))
            },
            'conclusion': f'Optimal pattern: pitch={optimal["pitch"]}, rhythm={optimal["rhythm"]}'
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
    
    experiment = CommunicationFrequencyExperiment()
    results = experiment.run()
    
    print("\n" + "="*70)
    print("COMMUNICATION FREQUENCY MATCHING EXPERIMENT RESULTS")
    print("="*70)
    print(f"Mean comprehension: {results['statistics']['mean_comprehension']:.3f}")
    print(f"Optimal pattern: {results['optimal_pattern']}")
    print(f"Conclusion: {results['conclusion']}")


if __name__ == "__main__":
    main()

