# CadQuery Code Generator - ML Project

## 🎯 Project Overview

This is a **Machine Learning (ML) project** that aims to create an AI model that can generate **CadQuery code** from images. Think of it like teaching a computer to look at a picture of a 3D object and write the code to create that object.

### What is CadQuery?
- **CadQuery** is a Python library for creating 3D CAD (Computer-Aided Design) models
- It's like having a programming language to build 3D objects
- Example: `cq.Workplane("XY").box(10, 20, 5)` creates a rectangular box

### What is Machine Learning?
- **ML** is teaching computers to learn patterns from data
- Instead of writing rules manually, we show the computer many examples
- The computer learns to make predictions based on what it has seen

## 📁 Project Structure

```
mecagent/
├── 📄 README.md                    # This file - project documentation
├── 📄 pyproject.toml              # Project dependencies and configuration
├── 📄 good_luck.ipynb             # Main Jupyter notebook with instructions
├── 📁 Metrics/                    # Evaluation tools
│   ├── 📄 __init__.py
│   ├── 📄 valid_syntax_rate.py    # Checks if generated code runs without errors
│   └── 📄 best_iou.py             # Compares 3D shapes for similarity
├── 📁 models/                     # ML models (to be created)
│   ├── 📄 __init__.py
│   ├── 📄 baseline_model.py       # Simple starting model
│   └── 📄 enhanced_model.py       # Improved model
├── 📁 data/                       # Dataset handling (to be created)
│   ├── 📄 __init__.py
│   ├── 📄 dataset_loader.py       # Loads the 147K image-code pairs
│   └── 📄 preprocessing.py        # Prepares data for training
├── 📁 training/                   # Training scripts (to be created)
│   ├── 📄 __init__.py
│   ├── 📄 train_baseline.py       # Trains the simple model
│   └── 📄 train_enhanced.py       # Trains the improved model
├── 📁 evaluation/                 # Evaluation scripts (to be created)
│   ├── 📄 __init__.py
│   ├── 📄 evaluate_baseline.py    # Tests the simple model
│   └── 📄 evaluate_enhanced.py    # Tests the improved model
└── 📁 utils/                      # Helper functions (to be created)
    ├── 📄 __init__.py
    ├── 📄 visualization.py        # Plot results and examples
    └── 📄 config.py              # Configuration settings
```

## 🎯 The Task (Step by Step)

### 1. **Load the Dataset**
- We have **147,000 pairs** of images and CadQuery code
- Each pair: Image of a 3D object + the code that creates it
- Goal: Learn the relationship between images and code

### 2. **Create a Baseline Model**
- Start with a **simple model** (baseline = starting point)
- This gives us a reference to compare against
- Like having a "before" picture to show improvement

### 3. **Enhance the Model**
- Make the model **better** using various techniques
- Could be: bigger model, better training, different architecture
- Goal: Get higher scores than the baseline

### 4. **Evaluate Both Models**
- Use two metrics to measure success:
  - **Valid Syntax Rate**: Does the generated code run without errors?
  - **Best IOU**: How similar are the 3D shapes created by the code?

### 5. **Explain Your Choices**
- Document what you tried and why
- Explain what worked and what didn't
- Identify potential problems (bottlenecks)

## 📊 Evaluation Metrics Explained

### 1. Valid Syntax Rate (VSR)
- **What it measures**: Percentage of generated code that runs without errors
- **Example**: 
  - ✅ Good: `result = cq.Workplane("XY").box(10, 10, 10)`
  - ❌ Bad: `result = cq.Workplane("XY").box(10, 10,` (missing parenthesis)
- **Goal**: Get as close to 100% as possible

### 2. Best IOU (Intersection over Union)
- **What it measures**: How similar two 3D shapes are
- **Range**: 0.0 (completely different) to 1.0 (identical)
- **Example**: 
  - IOU = 0.8 means 80% similarity between shapes
  - IOU = 0.3 means only 30% similarity
- **Goal**: Get as close to 1.0 as possible

## 🚀 Getting Started

### Prerequisites
1. **Python 3.11+** installed
2. **uv** package manager (for dependency management)
3. Basic understanding of Python

### Setup Steps
1. **Install uv** (if not already done):
   ```bash
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Activate virtual environment**:
   ```bash
   .venv\Scripts\Activate.ps1
   ```

4. **Open the notebook**:
   ```bash
   jupyter notebook good_luck.ipynb
   ```

## 🧠 Machine Learning Concepts for Beginners

### What is Training?
- **Training** = Teaching the model by showing it examples
- Like teaching a child: show many pictures of cats, they learn to recognize cats
- We show the model: "This image → produces this code"

### What is a Model?
- **Model** = The "brain" that learns patterns
- Takes input (image) and produces output (code)
- Gets better with more training data

### What is Evaluation?
- **Evaluation** = Testing how well the model works
- Use data the model hasn't seen before
- Measure performance with metrics (VSR, IOU)

## 🎨 Creative Freedom

**You can do ANYTHING you want!** This is your chance to be creative:

- Try different model architectures
- Experiment with training techniques
- Use pre-trained models
- Create custom loss functions
- Try data augmentation
- Use different optimizers

## 💡 Tips for Beginners

1. **Start Simple**: Don't try to build the perfect model first
2. **Iterate**: Build → Test → Improve → Repeat
3. **Document**: Write down what you try and why
4. **Compare**: Always compare against your baseline
5. **Ask Questions**: If something doesn't work, try to understand why

## 🔧 Technical Requirements

- **Dataset**: 147K image-code pairs from HuggingFace
- **Framework**: Any ML framework (PyTorch, TensorFlow, etc.)
- **Hardware**: CPU is fine, GPU helps but not required
- **Time**: Focus on relative improvement, not absolute performance

## 📈 Success Criteria

- **Baseline Model**: Simple model that works
- **Enhanced Model**: Better than baseline
- **Documentation**: Clear explanation of choices
- **Analysis**: Understanding of bottlenecks and improvements

Remember: **The journey matters more than the destination!** Focus on learning and experimentation rather than just getting the highest scores.