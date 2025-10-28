#!/usr/bin/env python
"""
Run All Validation Experiments

Convenience script to run all experiments sequentially.
"""

import sys
import subprocess
import logging
from pathlib import Path
import json


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


EXPERIMENTS = [
    ('Spectral Clustering', 'experiments/spectral_clustering/experiment.py'),
    ('Frequency-Dependent Recall', 'experiments/frequency_recall/experiment.py'),
    ('Golden Ratio Stability', 'experiments/golden_ratio/experiment.py'),
    ('Communication Frequency Matching', 'experiments/communication/experiment.py')
]


def run_experiment(name: str, script_path: str) -> bool:
    """
    Run a single experiment.
    
    Args:
        name: Experiment name
        script_path: Path to experiment script
        
    Returns:
        True if successful
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Running: {name}")
    logger.info(f"Script: {script_path}")
    logger.info('='*70 + "\n")
    
    if not Path(script_path).exists():
        logger.error(f"Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info(f"✓ {name} completed successfully")
            logger.info(result.stdout)
            return True
        else:
            logger.error(f"✗ {name} failed")
            logger.error(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {name} timed out")
        return False
    except Exception as e:
        logger.error(f"✗ {name} error: {e}")
        return False


def main():
    """Run all experiments."""
    print("\n" + "="*70)
    print("Running Validation Experiments")
    print("="*70)
    print("\nThis will run all 4 validation experiments sequentially.\n")
    
    results = {}
    
    for name, script_path in EXPERIMENTS:
        success = run_experiment(name, script_path)
        results[name] = success
        print()  # Blank line between experiments
    
    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:7} - {name}")
    
    print("\n" + "="*70)
    print(f"Results: {passed}/{total} experiments passed")
    print("="*70 + "\n")
    
    # Save summary
    summary_file = Path("experiments/results_summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump({
            'total': total,
            'passed': passed,
            'results': results
        }, f, indent=2)
    
    logger.info(f"Summary saved to {summary_file}")
    
    if passed == total:
        print("🎉 All experiments completed successfully!")
        sys.exit(0)
    else:
        print(f"⚠️  {total - passed} experiment(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

