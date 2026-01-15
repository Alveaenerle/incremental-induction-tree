# Incremental Induction Tree

A Python implementation of an incremental decision tree learning algorithm that can build and update decision trees incrementally from streaming data. This project demonstrates both batch learning and online learning capabilities, with comprehensive experimental comparisons against scikit-learn's standard decision tree classifier.

## Overview

This project implements an **Incremental Tree** algorithm that builds decision trees from data that arrives in batches or as a stream. The key innovation is the ability to update an existing tree with new samples without requiring complete retraining, making it ideal for scenarios with evolving datasets.

### Key Features

- **Incremental Learning**: Update existing trees with new data without full retraining
- **Batch Learning**: Build trees from complete datasets
- **Multiple Dataset Support**: Built-in loaders for IRIS, Australian Weather, and Airlines datasets
- **Experimental Framework**: Comprehensive test suite comparing quality, performance, and stability
- **Visualization**: Automatic generation of comparison charts and performance metrics

## Project Structure

```
.
├── main.py                          # Main entry point running all experiments
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── data/                            # Dataset directory
│   ├── airlines.csv
│   └── weatherAUS.csv
├── results/                         # Generated experiment results and visualizations
├── src/
│   ├── algorithms/
│   │   └── incremental_tree.py     # Core incremental tree implementation
│   ├── core/
│   │   ├── node.py                 # Tree node representation
│   │   └── stats.py                # Statistical utilities for split evaluation
│   ├── experiments/
│   │   └── suite.py                # Experimental framework and benchmarks
│   ├── utils/
│   │   ├── data_loader.py          # Dataset loading utilities
│   │   └── split_utils.py          # Feature splitting utilities
│   └── tests/
│       ├── test_inc.py             # Tests for incremental tree
│       ├── test_main.py            # Integration tests
│       ├── test_node.py            # Node structure tests
│       ├── test_split_utils.py     # Split utility tests
│       └── test_stats.py           # Statistics tests
```

## Installation

### Prerequisites
- Python 3.7+

### Setup

1. Clone or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

Execute the full experimental suite on all datasets:

```bash
python main.py
```

This will:
- Load IRIS, Australian Weather, and Airlines datasets
- Train the incremental tree on each dataset
- Compare against scikit-learn's DecisionTreeClassifier
- Generate performance metrics and visualizations
- Save results to the `results/` directory

### Using the Incremental Tree in Code

```python
from src.algorithms.incremental_tree import IncrementalTree
from src.utils.data_loader import DataLoader

# Load data
loader = DataLoader(bins=5)
train_data = loader.load_iris()

# Create and train tree
tree = IncrementalTree()
tree.fit(train_data)

# Make predictions
prediction = tree.predict(sample_row)

# Update tree with new data
tree.update(new_sample)
```

### Running Tests

```bash
pytest src/tests/
```

## Algorithm Details

### IncrementalTree

The `IncrementalTree` class implements a decision tree that supports:

- **`fit(data)`**: Build a tree from a batch of samples
- **`predict(row)`**: Classify a single sample
- **`update(sample)`**: Incrementally add a new sample to the tree

The algorithm works by:
1. Computing statistics (entropy, information gain) for feature splits
2. Recursively splitting nodes based on best information gain
3. Storing samples at leaf nodes for incremental updates
4. Re-evaluating splits when new data arrives

### Node Structure

Each node in the tree contains:
- **`statistics`**: Statistical information for split evaluation
- **`feature`**: The feature used for splitting (None if leaf)
- **`children`**: Dictionary mapping feature values to child nodes
- **`output`**: Predicted class (majority class at node)
- **`samples`**: Training samples stored at leaf nodes

## Datasets

The project includes loaders for three datasets:

1. **IRIS**: Classic iris flower classification (150 samples, 4 features)
2. **WEATHER**: Australian weather dataset (discretized continuous features)
3. **AIRLINES**: Airlines dataset (preprocessed for classification)

Data is automatically discretized into configurable bins for the decision tree algorithm.

## Experimental Framework

The `ExperimentSuite` class provides three types of experiments:

### 1. Quality Comparison
Compares classification accuracy between:
- Incremental tree (batch mode)
- Incremental tree (online mode with sequential updates)
- scikit-learn DecisionTreeClassifier

### 2. Time Performance
Benchmarks:
- Training time for batch learning
- Update time per sample for incremental learning
- Prediction speed

### 3. Structure Stability
Analyzes:
- Tree depth evolution
- Number of nodes over time
- Tree structure changes with incremental updates

## Results

Results are saved to the `results/` directory and include:
- CSV files with detailed metrics
- PNG visualizations comparing algorithms
- Performance graphs and charts

## Dependencies

Key libraries used:
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **scikit-learn**: DecisionTreeClassifier for comparison
- **matplotlib**: Visualization
- **pytest**: Testing framework

See `requirements.txt` for complete dependency list with versions.

## Testing

Unit tests are provided for all major components:

```bash
# Run all tests
pytest src/tests/ -v

# Run specific test
pytest src/tests/test_inc.py -v
```

## Performance Characteristics

- **Training**: O(n log n) for initial tree building
- **Prediction**: O(depth) per sample
- **Incremental Update**: O(depth) for single sample updates
- **Memory**: O(nodes) for tree storage, O(samples) at leaf nodes

## Future Improvements

- Support for regression tasks
- Handling of missing values
- Pruning strategies
- Multiclass distribution tracking
- Distributed incremental learning

## License

This is an educational/research project.

## Author

Developed as part of incremental learning research.
