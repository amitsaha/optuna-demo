# Optuna Hyperparameter Optimization Demos

This repository contains demonstrations of using [Optuna](https://optuna.readthedocs.io/en/stable/index.html) for hyperparameter tuning with XGBoost on the UCI Adult Income dataset.

## Overview

The demos showcase:
1. **Single-Objective Optimization**: Optimizing XGBoost hyperparameters to maximize classification accuracy
2. **Multi-Objective Optimization**: Finding optimal trade-offs between classification accuracy and model complexity (number of trees)

## Features

- 🎯 Interactive Streamlit web applications
- 📊 Real-time visualization of optimization progress
- 🔍 Parameter importance analysis
- 📈 Pareto frontier visualization for multi-objective optimization
- 🧪 Test set evaluation

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/amitsaha/optuna-demo.git
cd optuna-demo
```

2. Create a virtual environment 

Using conda:

``bash
conda create -n optuna-demo
```

Activate it:

```bash
conda activate optuna-demo
```

3. Install `pip`

```bash
conda install pip
```

``bash
4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Command-Line Demos

#### Single-Objective Optimization
```bash
python single_objective_demo.py
```


#### Single-Objective Optimization App
```bash
streamlit run streamlit_demo.py
```

Then open your browser to the URL displayed (typically http://localhost:8501).

Features:
- Adjust hyperparameter search ranges
- Set the number of optimization trials
- Visualize optimization history
- View parameter importance
- Evaluate the best model on the test set

This will run a single-objective optimization study with 50 trials, optimizing for maximum accuracy.

#### Multi-Objective Optimization
```bash
python multi_objective_demo.py
```

This will run a multi-objective optimization study with 50 trials, finding the Pareto frontier of solutions that balance accuracy and model complexity.

### Running Streamlit Web Apps


#### Multi-Objective Optimization App
```bash
streamlit run streamlit_multi_objective.py
```

Then open your browser to the URL displayed (typically http://localhost:8501).

Features:
- Adjust hyperparameter search ranges
- Set the number of optimization trials
- Visualize the Pareto frontier
- Explore optimal trade-off solutions
- Select and evaluate specific solutions on the test set

## Dataset

The demos use the **UCI Adult Income** dataset, which is automatically downloaded via the `ucimlrepo` package. This dataset contains census data and is used to predict whether an individual's income exceeds $50K/year based on various demographic features.

- **Features**: 14 (age, workclass, education, marital-status, occupation, etc.)
- **Target**: Binary classification (income >50K or ≤50K)
- **Samples**: ~48,000

**Note**: If the UCI ML Repository is unavailable, the demos will automatically use a synthetic dataset with similar characteristics for demonstration purposes.

## Hyperparameters Tuned

The following XGBoost hyperparameters are optimized:

- `max_depth`: Maximum tree depth
- `learning_rate`: Step size shrinkage
- `n_estimators`: Number of boosting rounds

## Key Concepts

### Single-Objective Optimization

In single-objective optimization, we aim to maximize a single metric (accuracy). Optuna explores the hyperparameter space and finds the configuration that yields the best performance.

### Multi-Objective Optimization

In multi-objective optimization, we optimize for multiple competing objectives simultaneously:
1. **Maximize accuracy**: Better predictive performance
2. **Minimize complexity**: Fewer trees means faster inference and simpler model

The result is a Pareto frontier - a set of solutions where improving one objective would require sacrificing another. This allows you to choose the best trade-off for your specific needs.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## References

- [Optuna Documentation](https://optuna.readthedocs.io/en/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [UCI Adult Income Dataset](https://archive.ics.uci.edu/dataset/2/adult)
