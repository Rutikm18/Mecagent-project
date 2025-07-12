"""
Evaluation script for the baseline CadQuery code generator.
This script tests the baseline model using the provided evaluation metrics.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config import MAX_EVAL_SAMPLES, IOU_PITCH
from data.dataset_loader import CadQueryDatasetLoader
from models.baseline_model import BaselineCadQueryGenerator
from Metrics.valid_syntax_rate import evaluate_syntax_rate_simple
from Metrics.best_iou import get_iou_best

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineEvaluator:
    """
    Evaluator for the baseline CadQuery code generator.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.dataset_loader = CadQueryDatasetLoader()
        self.model = BaselineCadQueryGenerator()
        
        logger.info("Baseline evaluator initialized")
    
    def evaluate_model(self, max_samples: Optional[int] = None) -> Dict:
        """
        Evaluate the baseline model on the test dataset.
        
        Args:
            max_samples: Maximum number of samples to evaluate (for testing)
            
        Returns:
            Dictionary with evaluation results
        """
        max_samples = max_samples or MAX_EVAL_SAMPLES
        
        logger.info(f"Starting evaluation with max {max_samples} samples")
        start_time = time.time()
        
        # Load test data
        logger.info("Loading test dataset...")
        test_data = self.dataset_loader.get_data_for_training(
            split='test', 
            max_samples=max_samples
        )
        
        logger.info(f"Loaded {len(test_data)} test samples")
        
        # Generate predictions
        logger.info("Generating predictions...")
        images = [sample['image'] for sample in test_data]
        expected_codes = [sample['code'] for sample in test_data]
        
        predicted_codes = self.model.predict_batch(images)
        
        # Create dictionaries for evaluation
        codes_dict = {f"sample_{i}": code for i, code in enumerate(predicted_codes)}
        expected_dict = {f"sample_{i}": code for i, code in enumerate(expected_codes)}
        
        # Evaluate Valid Syntax Rate
        logger.info("Evaluating Valid Syntax Rate...")
        vsr = evaluate_syntax_rate_simple(codes_dict)
        
        # Evaluate IOU (on a subset for speed)
        logger.info("Evaluating IOU...")
        iou_scores = []
        iou_subset_size = min(100, len(predicted_codes))  # Limit IOU evaluation for speed
        
        for i in range(iou_subset_size):
            try:
                iou = get_iou_best(predicted_codes[i], expected_codes[i])
                iou_scores.append(iou)
                
                if i % 20 == 0:
                    logger.info(f"IOU evaluation progress: {i}/{iou_subset_size}")
                    
            except Exception as e:
                logger.warning(f"IOU evaluation failed for sample {i}: {e}")
                iou_scores.append(0.0)
        
        avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
        
        # Calculate additional metrics
        total_time = time.time() - start_time
        avg_time_per_sample = total_time / len(test_data)
        
        # Compile results
        results = {
            'valid_syntax_rate': vsr,
            'average_iou': avg_iou,
            'iou_scores': iou_scores,
            'total_samples': len(test_data),
            'iou_evaluated_samples': len(iou_scores),
            'total_evaluation_time': total_time,
            'avg_time_per_sample': avg_time_per_sample,
            'sample_predictions': list(zip(predicted_codes[:5], expected_codes[:5]))  # First 5 samples
        }
        
        logger.info("Evaluation completed")
        logger.info(f"Valid Syntax Rate: {vsr:.3f}")
        logger.info(f"Average IOU: {avg_iou:.3f}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        
        return results
    
    def print_detailed_results(self, results: Dict) -> None:
        """
        Print detailed evaluation results.
        
        Args:
            results: Evaluation results dictionary
        """
        print("\n" + "="*60)
        print("BASELINE MODEL EVALUATION RESULTS")
        print("="*60)
        
        print(f"Valid Syntax Rate: {results['valid_syntax_rate']:.1%}")
        print(f"Average IOU: {results['average_iou']:.3f}")
        print(f"Total samples evaluated: {results['total_samples']}")
        print(f"IOU samples evaluated: {results['iou_evaluated_samples']}")
        print(f"Total evaluation time: {results['total_evaluation_time']:.2f} seconds")
        print(f"Average time per sample: {results['avg_time_per_sample']:.3f} seconds")
        
        print("\nSample Predictions (first 3):")
        print("-" * 40)
        for i, (predicted, expected) in enumerate(results['sample_predictions'][:3]):
            print(f"Sample {i+1}:")
            print(f"Predicted: {predicted[:100]}...")
            print(f"Expected:  {expected[:100]}...")
            print()
        
        print("IOU Score Distribution:")
        print("-" * 40)
        iou_scores = results['iou_scores']
        if iou_scores:
            print(f"Min IOU: {min(iou_scores):.3f}")
            print(f"Max IOU: {max(iou_scores):.3f}")
            print(f"Median IOU: {sorted(iou_scores)[len(iou_scores)//2]:.3f}")
            
            # Count scores in different ranges
            excellent = sum(1 for score in iou_scores if score >= 0.8)
            good = sum(1 for score in iou_scores if 0.6 <= score < 0.8)
            fair = sum(1 for score in iou_scores if 0.4 <= score < 0.6)
            poor = sum(1 for score in iou_scores if score < 0.4)
            
            print(f"Excellent (≥0.8): {excellent} ({excellent/len(iou_scores):.1%})")
            print(f"Good (0.6-0.8): {good} ({good/len(iou_scores):.1%})")
            print(f"Fair (0.4-0.6): {fair} ({fair/len(iou_scores):.1%})")
            print(f"Poor (<0.4): {poor} ({poor/len(iou_scores):.1%})")
        
        print("="*60)
    
    def save_results(self, results: Dict, filename: str = "baseline_evaluation_results.json") -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            results: Evaluation results dictionary
            filename: Name of the file to save results
        """
        import json
        from datetime import datetime
        
        # Add timestamp
        results['evaluation_timestamp'] = datetime.now().isoformat()
        results['model_name'] = 'baseline_cadquery_generator'
        
        # Remove non-serializable items
        results_copy = results.copy()
        if 'sample_predictions' in results_copy:
            del results_copy['sample_predictions']  # Remove for JSON serialization
        
        filepath = Path(__file__).parent / filename
        
        with open(filepath, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")

def main():
    """Run the baseline evaluation."""
    print("Baseline Model Evaluation")
    print("=" * 40)
    
    try:
        # Initialize evaluator
        evaluator = BaselineEvaluator()
        
        # Run evaluation
        results = evaluator.evaluate_model(max_samples=50)  # Start with small number for testing
        
        # Print results
        evaluator.print_detailed_results(results)
        
        # Save results
        evaluator.save_results(results)
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main() 