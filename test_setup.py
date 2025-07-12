"""
Test script to verify that the basic project setup is working.
Run this to check if everything is properly configured.
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import cadquery as cq
        print("✓ CadQuery imported successfully")
    except ImportError as e:
        print(f"✗ CadQuery import failed: {e}")
        return False
    
    try:
        from datasets import load_dataset
        print("✓ Datasets imported successfully")
    except ImportError as e:
        print(f"✗ Datasets import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import trimesh
        print("✓ Trimesh imported successfully")
    except ImportError as e:
        print(f"✗ Trimesh import failed: {e}")
        return False
    
    return True

def test_project_structure():
    """Test if the project structure is correct."""
    print("\nTesting project structure...")
    
    project_root = Path(__file__).parent
    required_dirs = ['models', 'data', 'training', 'evaluation', 'utils', 'Metrics']
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✓ Directory {dir_name} exists")
        else:
            print(f"✗ Directory {dir_name} missing")
            return False
    
    return True

def test_config():
    """Test if the configuration can be loaded."""
    print("\nTesting configuration...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        from utils.config import PROJECT_ROOT, DATASET_NAME
        print(f"✓ Configuration loaded successfully")
        print(f"  Project root: {PROJECT_ROOT}")
        print(f"  Dataset name: {DATASET_NAME}")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_baseline_model():
    """Test if the baseline model can be created."""
    print("\nTesting baseline model...")
    
    try:
        from models.baseline_model import BaselineCadQueryGenerator
        model = BaselineCadQueryGenerator()
        print("✓ Baseline model created successfully")
        
        # Test prediction
        fake_image = "test_image"
        code = model.predict(fake_image)
        print(f"✓ Model prediction works: {len(code)} characters generated")
        
        return True
    except Exception as e:
        print(f"✗ Baseline model test failed: {e}")
        return False

def test_metrics():
    """Test if the evaluation metrics work."""
    print("\nTesting evaluation metrics...")
    
    try:
        from Metrics.valid_syntax_rate import evaluate_syntax_rate_simple
        from Metrics.best_iou import get_iou_best
        
        # Test VSR
        test_codes = {
            "test1": "result = cq.Workplane('XY').box(10, 10, 10)",
            "test2": "result = cq.Workplane('XY').cylinder(10, 5)"
        }
        vsr = evaluate_syntax_rate_simple(test_codes)
        print(f"✓ Valid Syntax Rate works: {vsr}")
        
        # Test IOU
        code1 = "result = cq.Workplane('XY').box(10, 10, 10)"
        code2 = "result = cq.Workplane('XY').box(10, 10, 10)"
        iou = get_iou_best(code1, code2)
        print(f"✓ IOU calculation works: {iou}")
        
        return True
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("CadQuery Code Generator - Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_project_structure,
        test_config,
        test_baseline_model,
        test_metrics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready to go.")
        print("\nNext steps:")
        print("1. Open good_luck.ipynb in Jupyter")
        print("2. Start with the baseline model")
        print("3. Experiment and improve!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("- Make sure you're in the virtual environment")
        print("- Run 'uv sync' to install dependencies")
        print("- Check that all directories were created")

if __name__ == "__main__":
    main() 