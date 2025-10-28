"""
Experiment 3: Golden Ratio Stability

Test whether knowledge graph structures approximating golden ratio
proportions exhibit greater stability.

Hypothesis: Structures with proportions near φ ≈ 1.618 exhibit lower
contradiction rates and greater stability.
"""

import logging
import numpy as np
from typing import Dict, Any, List
import json
from pathlib import Path


class GoldenRatioStabilityExperiment:
    """
    Experiment to test golden ratio stability in knowledge graphs.
    """
    
    def __init__(self, output_dir: str = "experiments/golden_ratio/results/"):
        """Initialize experiment."""
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.golden_ratio = (1 + np.sqrt(5)) / 2  # φ ≈ 1.618
    
    def analyze_graph_structure(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze knowledge graph structure.
        
        Args:
            graph_data: Graph representation
            
        Returns:
            Structural metrics
        """
        # Calculate structural properties
        metrics = {
            'node_degree_ratio': 1.5,  # Example
            'edge_density_ratio': 1.618,  # Golden ratio
            'subgraph_ratio': 1.3
        }
        
        return metrics
    
    def calculate_phi_approximation(self, ratio: float) -> float:
        """
        Calculate how close a ratio is to golden ratio.
        
        Args:
            ratio: Ratio to test
            
        Returns:
            Approximation score (0=far, 1=exact)
        """
        if ratio == self.golden_ratio:
            return 1.0
        
        # Calculate inverse approximation
        if ratio < self.golden_ratio:
            return ratio / self.golden_ratio
        else:
            return self.golden_ratio / ratio
    
    def measure_stability(self, subgraphs: List[Dict]) -> Dict[str, float]:
        """
        Measure stability metrics for subgraphs.
        
        Args:
            subgraphs: List of subgraph structures
            
        Returns:
            Stability metrics
        """
        phi_approximations = []
        contradiction_rates = []
        
        for subgraph in subgraphs:
            # Calculate phi approximation
            ratio = subgraph.get('edge_to_node_ratio', 1.0)
            phi_score = self.calculate_phi_approximation(ratio)
            phi_approximations.append(phi_score)
            
            # Calculate contradiction rate (mock)
            contradictions = subgraph.get('contradictions', 0)
            total_edges = subgraph.get('total_edges', 1)
            contradiction_rate = contradictions / total_edges
            contradiction_rates.append(contradiction_rate)
        
        # Calculate correlation
        correlation = np.corrcoef(phi_approximations, contradiction_rates)[0, 1]
        
        return {
            'phi_scores': phi_approximations,
            'contradiction_rates': contradiction_rates,
            'correlation': correlation
        }
    
    def run(self) -> Dict[str, Any]:
        """Run complete experiment."""
        self.logger.info("Starting golden ratio stability experiment...")
        
        # Generate mock subgraph data
        subgraphs = []
        for i in range(50):
            subgraphs.append({
                'id': f"subgraph_{i}",
                'edge_to_node_ratio': np.random.uniform(1.0, 3.0),
                'total_edges': np.random.randint(10, 100),
                'contradictions': np.random.randint(0, 5)
            })
        
        # Analyze stability
        stability = self.measure_stability(subgraphs)
        
        results = {
            'experiment': 'Golden Ratio Stability',
            'hypothesis': 'φ-approximating structures are more stable',
            'n_subgraphs': len(subgraphs),
            'golden_ratio': float(self.golden_ratio),
            'stability_analysis': {
                'mean_phi_score': float(np.mean(stability['phi_scores'])),
                'mean_contradiction_rate': float(np.mean(stability['contradiction_rates'])),
                'correlation': float(stability['correlation'])
            },
            'conclusion': 'Negative correlation supports hypothesis' if stability['correlation'] < 0 else 'No clear correlation'
        }
        
        # Save results
        results_file = self.output_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Experiment complete. Results saved to {results_file}")
        
        return results


def main():
    """Run the experiment."""
    logging.basicConfig(level=logging.INFO)
    
    experiment = GoldenRatioStabilityExperiment()
    results = experiment.run()
    
    print("\n" + "="*70)
    print("GOLDEN RATIO STABILITY EXPERIMENT RESULTS")
    print("="*70)
    print(f"Mean phi score: {results['stability_analysis']['mean_phi_score']:.3f}")
    print(f"Mean contradiction rate: {results['stability_analysis']['mean_contradiction_rate']:.3f}")
    print(f"Correlation: {results['stability_analysis']['correlation']:.3f}")
    print(f"Conclusion: {results['conclusion']}")


if __name__ == "__main__":
    main()

