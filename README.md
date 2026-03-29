# Optimizer Stability Research

## Overview

This project presents an experimental study on the stability and robustness of deep learning optimizers under challenging training conditions.

We analyze how different optimization methods behave when exposed to:
- Label noise (data corruption)
- Numerical precision constraints (float32 vs float64)

The goal is to understand how optimizer choice affects convergence, accuracy, and generalization in noisy environments.

---

## Optimizers Evaluated

- **SGD (Stochastic Gradient Descent)**  
  Baseline optimization method

- **Adam**  
  Adaptive learning rate optimizer

- **SAM (Sharpness-Aware Minimization)**  
  Optimizer designed to improve generalization by seeking flat minima

---

## Experimental Factors

| Factor            | Values                     |
|------------------|---------------------------|
| Label Noise      | 0%, 10%, 30%              |
| Precision        | float32, float64          |
| Optimizers       | SGD, Adam, SAM (SGD/Adam) |
| Seeds            | 0, 1, 2, 3, 4             |
| Epochs           | 3–20                      |

---

## Dataset

- **MNIST handwritten digits**
- 60,000 training samples
- 10,000 test samples
- 10 classes

Artificial label noise is introduced by randomly flipping labels based on the selected noise level.

---

## Methodology

The model is trained using standard cross-entropy loss:

L(θ) = - (1/N) Σ y log(ŷ)

For SAM, weights are perturbed before update:

ε = ρ * (∇L / ||∇L||)

θ(t+1) = θ(t) - η ∇L(θ + ε)

Where:
- ρ = neighborhood size (e.g. 0.05)
- η = learning rate

---

## Key Results

- SAM demonstrates strong robustness under high label noise
- Adam converges faster but is more sensitive to noisy labels
- SGD is stable but slower compared to adaptive methods
- float64 improves numerical stability slightly but increases computation cost

---

## Project Structure


optimizer-stability/
│
├── src/
│ ├── train.py
│ ├── analyze.py
│ └── models/
│
├── results/
├── run_grid.ps1
├── run_grid_light.ps1
├── requirements.txt
├── README.md
└── .gitignore


---

## Installation

### 1. Clone the repository


git clone https://github.com/kamalzada37/optimizer-stability-clean.git

cd optimizer-stability-clean


### 2. Create virtual environment


python -m venv .venv
.venv\Scripts\Activate.ps1


### 3. Install dependencies


pip install --index-url https://download.pytorch.org/whl/cpu
 torch torchvision
pip install -r requirements.txt


---

## Usage

### Run training


python -m src.train --optimizer adam --lr 0.001 --noise 0.1 --precision float32 --seed 0 --epochs 3 --outdir results/light --dataset mnist


### Run analysis


python -m src.analyze --indir results/light


---

## Outputs

Generated outputs include:

- `summary.csv` — aggregated accuracy results
- `precision_gap.csv` — float32 vs float64 comparison
- Accuracy plots across noise levels
- Training curves over epochs

---

## Visualization Examples

- Accuracy vs Noise Level
- Training Curves
- Precision Comparison

(Add plots here from results/ folder)

---

## Type

Academic Research Project (Bachelor Level)

---

## Citation

If you use this work:

Kamal Zada, M. & ParsaKarimi, Z. (2025)  
"Numerical Optimization in Machine Learning: Stability and Robustness of Optimizers Under Noise and Precision Constraints"

---

## License

MIT License

---

## Acknowledgments

- PyTorch
- Torchvision
- MNIST dataset (Yann LeCun et al.)
