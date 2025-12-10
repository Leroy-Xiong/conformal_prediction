# Conformal Prediction Experiments

This repository contains implementations and experiments demonstrating the key properties of Conformal Prediction (CP) methods for reliable uncertainty quantification in machine learning.

## Project Structure

```
├── main.py                  # Main entry point to run all experiments
├── modules/
│   ├── finite_sample.py     # Finite sample validity experiment
│   ├── distribution_free.py # Distribution-free guarantee experiment
│   ├── model_free.py        # Model agnosticism experiment
│   ├── model_comparison.py  # Comparison with other methods
│   └── presentation.py      # Experiments for presentations
├── report/
│   ├── report.tex           # Technical report in LaTeX
│   ├── report.pdf           # Compiled PDF version of the report
│   └── figures/             # Figures used in the report
└── results/                 # Output directory for experiment results
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Leroy-Xiong/conformal_prediction.git
cd conformal_prediction

# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate math5472
```

## Usage

Run all experiments:

```bash
python main.py
```

This will execute all experiments and save results to the `./results/` directory.