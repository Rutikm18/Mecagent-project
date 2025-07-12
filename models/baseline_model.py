"""
Baseline model for CadQuery code generation.
This is a simple starting point that you can improve upon.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config import BASELINE_MODEL_NAME, MODELS_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineCadQueryGenerator:
    """
    A simple baseline model for generating CadQuery code from images.
    
    This is a very basic implementation that serves as a starting point.
    It demonstrates the structure but doesn't actually learn from images.
    """
    
    def __init__(self, model_name: str = BASELINE_MODEL_NAME):
        """
        Initialize the baseline model.
        
        Args:
            model_name: Name for saving/loading the model
        """
        self.model_name = model_name
        self.model_path = MODELS_DIR / f"{model_name}.pkl"
        
        # Simple templates for different shapes
        self.templates = {
            'box': """
height = {height}
width = {width}
thickness = {thickness}

result = cq.Workplane("XY").box(height, width, thickness)
""",
            'cylinder': """
height = {height}
diameter = {diameter}

result = cq.Workplane("XY").cylinder(height, diameter/2)
""",
            'sphere': """
diameter = {diameter}

result = cq.Workplane("XY").sphere(diameter/2)
""",
            'complex_box': """
height = {height}
width = {width}
thickness = {thickness}
diameter = {diameter}
padding = {padding}

# make the base
result = (
    cq.Workplane("XY")
    .box(height, width, thickness)
    .faces(">Z")
    .workplane()
    .hole(diameter)
    .faces(">Z")
    .workplane()
    .rect(height - padding, width - padding, forConstruction=True)
    .vertices()
    .cboreHole(2.4, 4.4, 2.1)
)
"""
        }
        
        logger.info(f"Baseline model initialized: {model_name}")
    
    def predict(self, image, **kwargs) -> str:
        """
        Generate CadQuery code from an image.
        
        Args:
            image: Input image (not actually used in baseline)
            **kwargs: Additional parameters
            
        Returns:
            Generated CadQuery code as string
        """
        # This is a very simple baseline that just returns random templates
        # In a real model, you would analyze the image and generate appropriate code
        
        template_name = random.choice(list(self.templates.keys()))
        template = self.templates[template_name]
        
        # Generate random parameters
        params = {
            'height': random.uniform(20, 100),
            'width': random.uniform(20, 100),
            'thickness': random.uniform(5, 30),
            'diameter': random.uniform(10, 50),
            'padding': random.uniform(5, 20)
        }
        
        # Format the template with random parameters
        generated_code = template.format(**params)
        
        logger.info(f"Generated code using template: {template_name}")
        return generated_code.strip()
    
    def predict_batch(self, images: List, **kwargs) -> List[str]:
        """
        Generate CadQuery code for multiple images.
        
        Args:
            images: List of input images
            **kwargs: Additional parameters
            
        Returns:
            List of generated CadQuery code strings
        """
        results = []
        for i, image in enumerate(images):
            code = self.predict(image, **kwargs)
            results.append(code)
            
            if i % 100 == 0:
                logger.info(f"Processed {i} images...")
        
        return results
    
    def save(self, filepath: Optional[Path] = None) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model (optional)
        """
        if filepath is None:
            filepath = self.model_path
        
        # For this simple baseline, we just save the templates
        import pickle
        
        model_data = {
            'model_name': self.model_name,
            'templates': self.templates
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to: {filepath}")
    
    def load(self, filepath: Optional[Path] = None) -> None:
        """
        Load the model from disk.
        
        Args:
            filepath: Path to load the model from (optional)
        """
        if filepath is None:
            filepath = self.model_path
        
        if not filepath.exists():
            logger.warning(f"Model file not found: {filepath}")
            return
        
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_name = model_data['model_name']
        self.templates = model_data['templates']
        
        logger.info(f"Model loaded from: {filepath}")
    
    def evaluate_sample(self, image, expected_code: str) -> Dict:
        """
        Evaluate the model on a single sample.
        
        Args:
            image: Input image
            expected_code: Expected/ground truth code
            
        Returns:
            Dictionary with evaluation metrics
        """
        generated_code = self.predict(image)
        
        # Simple evaluation metrics
        metrics = {
            'generated_code': generated_code,
            'expected_code': expected_code,
            'code_length': len(generated_code),
            'has_cadquery_import': 'import cadquery' in generated_code.lower(),
            'has_workplane': 'workplane' in generated_code.lower(),
            'has_result': 'result' in generated_code.lower()
        }
        
        return metrics

def main():
    """Test the baseline model."""
    print("Testing Baseline CadQuery Generator")
    print("=" * 40)
    
    # Initialize model
    model = BaselineCadQueryGenerator()
    
    # Test single prediction
    print("Testing single prediction:")
    fake_image = "fake_image_data"  # In real usage, this would be an actual image
    code = model.predict(fake_image)
    print(f"Generated code:\n{code}")
    print()
    
    # Test batch prediction
    print("Testing batch prediction:")
    fake_images = ["image1", "image2", "image3"]
    codes = model.predict_batch(fake_images)
    for i, code in enumerate(codes):
        print(f"Image {i+1} code:\n{code[:100]}...")
        print()
    
    # Test evaluation
    print("Testing evaluation:")
    expected_code = "result = cq.Workplane('XY').box(10, 10, 10)"
    metrics = model.evaluate_sample(fake_image, expected_code)
    print(f"Evaluation metrics: {metrics}")
    print()
    
    # Test save/load
    print("Testing save/load:")
    model.save()
    new_model = BaselineCadQueryGenerator()
    new_model.load()
    print("Save/load test completed")

if __name__ == "__main__":
    main() 