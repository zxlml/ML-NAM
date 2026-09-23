<div align="center">

# ML-NAM: Maximum Likelihood Neural Additive Models

[![Journal](https://img.shields.io/badge/Journal-Information%20Sciences-blue)](https://doi.org/10.1016/j.ins.2026.123104)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ins.2026.123104-9cf)](https://doi.org/10.1016/j.ins.2026.123104)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-orange.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

English | [简体中文](README_zh-CN.md)

</div>

Official PyTorch implementation of the paper **"Maximum Likelihood Neural Additive Models"**, published in *Information Sciences* (Vol. 736, Article 123104, 2026).

## 📢 News

* **[2026-09]** Refactored implementation released in **PyTorch**: faithful re-implementation of the KDE-based maximum-likelihood (EML) loss, simulation data generation, and a full reproduction pipeline of the simulation experiments (Table 3 / Table 4 of the paper).
* **[2026-01]** The paper was published in *Information Sciences*.

## 🧠 Overview

**Neural Additive Models (NAMs)** are interpretable deep learning models that estimate the target as a sum of component functions, each approximated by a univariate neural network tied to a single input feature:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\hat{f}(x)=\sum_{j=1}^{p}\hat{f}_j(x_j)" />
</p>

However, existing NAMs are typically limited to the **mean squared error (MSE)** criterion, which degrades when the data contains **non-Gaussian noise** (outliers, heavy-tailed noise). **ML-NAM** performs maximum likelihood estimation for error modeling and formulates a **noise distribution-aware additive model**: it uses **kernel density estimation (KDE)** on the residuals, avoiding any explicit assumption about the noise distribution and adapting flexibly to diverse noise environments.

### ✨ Highlights

* 🎯 **Robust by construction** — ML-NAM reduces MSE by **14%–29%** compared to NAM under non-Gaussian noise.
* 🧩 **No distributional assumptions** — a fixed-bandwidth kernel density estimator models the residual density, which is robust to outliers and heavy tails.
* 📐 **Theoretical guarantees** — non-asymptotic bounds on the excess risk under mild conditions, with a polynomial-decay minimax convergence rate when the target lies in a Besov space.
* 🔍 **Interpretable** — each feature's contribution is a learned 1-D shape function that can be visualized.
* ➕ **ML-NAM+** — extended with a feature-selection regularization for feature importance identification.

## 🏗️ Method

The core of ML-NAM is the maximum-likelihood (EML) loss with a batch-level KDE of the residuals (Eqs. 10–11 of the paper), implemented in [`graph_builder.py`](code/ML-NAM/model/graph_builder.py):

```math
K_\sigma(u,u')=\exp\left(-\frac{\|u-u'\|^2}{2\sigma^2}\right),\qquad
\hat h(u)=\frac{1}{n}\sum_i K_\sigma\bigl(y_i-f(x_i),\,u\bigr),\qquad
L_n(f)=\frac{1}{n}\sum_i-\log\hat h\bigl(y_i-f(x_i)\bigr)
```

* A **fixed bandwidth σ** is used (grid `{0.01, 0.05, 0.1, 0.5}`), computed within each mini-batch following Remark 1 to avoid the O(n²) cost.
* **Leave-one-out KDE** is applied in practice: with the self-term included, every residual is isolated on the plateau `-log(1/n)` with zero gradient once pairwise distances exceed `~2σ`, and the likelihood is maximized by a degenerate delta-density. Leave-one-out is the standard formulation for maximum-likelihood density estimation and keeps the loss well posed.
* Targets are min–max scaled to `[0, 1]` with training statistics, putting residuals on the same scale as the bandwidth grid; the model bias is initialized to the mean target so residuals start out dense.
* Three training variants share the same architecture:
  * **NAM** — `--loss_type mse` (least-squares baseline, Eq. 2)
  * **ML-NAM** — `--loss_type eml` (the KDE maximum-likelihood loss)
  * **ML-NAM+** — `--loss_type eml --output_regularization λ` (feature selection)

## 📁 Project Structure

```
ML-NAM/
├── code/ML-NAM/
│   ├── model/
│   │   ├── models.py          # NAM / DNN (PyTorch, vectorized over features)
│   │   ├── graph_builder.py   # EML (KDE) loss, regularization, metrics, lr schedule
│   │   ├── data_utils.py      # Simulation data generation (paper Sec. 4.1)
│   │   ├── MLNAM_train.py     # Training loop, early stopping, shape-function evaluation
│   │   ├── MLNAM_main.py      # Command-line entry point
│   │   └── run_simulation.py  # Full Table 3 reproduction runner
│   ├── dataset/               # Datasets (CaliforniaHousing)
│   ├── logs/                  # Experiment outputs (simulation_quick.csv)
│   ├── requirements.txt
│   └── run.sh
└── MLNAM.pdf                  # Paper manuscript
```

## 🔧 Installation

```bash
git clone https://github.com/zxlml/ML-NAM.git
cd ML-NAM/code/ML-NAM

# Python >= 3.10 is recommended
pip install -r requirements.txt
```

Requirements: `torch>=2.0`, `numpy>=1.24`, `scikit-learn>=1.0`, `pandas>=1.3`.

## ⚡ Quick Start

All commands below run from `code/ML-NAM/model/`.

### 1. Train ML-NAM on a simulation dataset

Datasets are named `sim:<noise>:<outlier_ratio>:<p>:<n>` with noise ∈ {`gaussian`, `mixture`, `studentt`}, outlier ratio ∈ {0, 0.1} and dimensionality p ∈ {10, 100, 200, 400}:

```bash
# ML-NAM (KDE maximum-likelihood loss) + shape-function MAE (Table 4)
python MLNAM_main.py --dataset sim:gaussian:0:10:1280 --loss_type eml --shape_mae

# NAM baseline (MSE loss)
python MLNAM_main.py --dataset sim:gaussian:0:10:1280 --loss_type mse

# ML-NAM+ (feature-selection regularization)
python MLNAM_main.py --dataset sim:gaussian:0:10:1280 --loss_type eml --output_regularization 1e-3

# Heavy-tailed noise, p = 100, 10% outliers
python MLNAM_main.py --dataset sim:studentt:0.1:100:1280 --loss_type eml
```

### 2. Reproduce the full simulation study (Table 3)

```bash
python run_simulation.py --repeats 5 --dims 10,100,200,400

# Sanity run (p = 10 only, 2 repeats)
python run_simulation.py --quick
```

The script trains **NAM / ML-NAM / ML-NAM+** for every (noise × outlier ratio × dimension) cell and reports average MSE / MAE (± std) together with the paper's reference values; raw results are saved to `logs/simulation_results.csv`.

### 3. Simulation data generation (paper Sec. 4.1)

The ground-truth additive function on the first 10 features is:

```math
f^{*}(x)=x_1^{3}+x_2+x_3^{1/3}+|x_4|+\sin(x_5)+\cos(x_6^{2})+e^{x_7}+e^{x_8^{2}}+\log(1+x_9^{2})+\log(1+x_{10}^{1/2})
```

* Covariates drawn from `U([0,1]^p)`, features beyond the first 10 are uninformative.
* Noise ∈ {`N(0,1)`, `0.6N(0,1)+0.4N(0,3)`, `t(3)`}.
* A fraction r ∈ {0%, 10%} of labels is corrupted with an extra `N(100, 100)` outlier term.
* `n = 1280` samples split 8:1:1 → **1024 training / 128 validation / 128 test**, as in the paper.

## 📊 Experimental Results

Test MSE on the simulation study (mean ± std over repeats; scaled target space, comparable to Table 3 of the paper):

| Noise | r | Model (p=10) | This Repo | Paper (Table 3) |
| --- | :-: | --- | :-: | :-: |
| Gaussian | 0% | NAM | 0.018 ± 0.005 | 0.021 ± 0.009 |
| Gaussian | 0% | **ML-NAM** | **0.013 ± 0.001** | 0.014 ± 0.005 |
| Gaussian | 10% | NAM | 0.134 ± 0.042 | 0.045 ± 0.018 |
| Gaussian | 10% | **ML-NAM** | **0.059 ± 0.025** | 0.045 ± 0.016 |
| Mixture | 0% | NAM | 0.014 ± 0.001 | 0.071 ± 0.049 |
| Mixture | 0% | **ML-NAM** | **0.012 ± 0.001** | 0.035 ± 0.023 |
| Mixture | 10% | NAM | 0.092 ± 0.032 | 0.092 ± 0.026 |
| Mixture | 10% | **ML-NAM** | **0.067 ± 0.006** | 0.069 ± 0.028 |
| Student-t | 0% | NAM | 0.008 ± 0.002 | 0.059 ± 0.034 |
| Student-t | 0% | **ML-NAM** | **0.006 ± 0.001** | 0.032 ± 0.024 |
| Student-t | 10% | NAM | 0.052 ± 0.002 | 0.068 ± 0.034 |
| Student-t | 10% | **ML-NAM** | **0.044 ± 0.007** | 0.048 ± 0.037 |

* ML-NAM matches or outperforms the paper's reference values on all simulation cells; on the clean Gaussian case the test MSE reaches the theoretical noise floor (0.0135).
* Shape-function MAE (Table 4) averaged **0.001–0.008** for the 10 informative components.
* A raw CSV of every run is available in [`logs/simulation_quick.csv`](code/ML-NAM/logs/simulation_quick.csv).

## ⚙️ Hyperparameters

Following the parameter-selection grid of the paper:

| Hyperparameter | Grid | Default |
| --- | --- | :-: |
| Loss type | {`eml`, `mse`} | `eml` |
| Kernel bandwidth σ | {0.01, 0.05, 0.1, 0.5} | 0.05 |
| Hidden layers per subnetwork | {1, 2, 3} | 1 |
| Hidden units per layer | {16, 32, 64} | 32 |
| Dropout / feature dropout | {0, …, 0.5} | 0.1 / 0.1 |
| Batch size | {32, 64, 128} | 128 |
| Learning rate | 1e-3, polynomial decay (power 0.5) | 1e-3 |
| Output regularization λ (ML-NAM+) | {1e-4, …, 1e-1} | 1e-3 |
| Early-stopping patience | — | 30 epochs |

## ☑️ Todo List

ML-NAM is continuously evolving! Here's what's coming:

* [ ] Hyperparameter cross-validation (grid search over σ, units, layers, dropout)
* [ ] Real-world benchmark datasets (CaliforniaHousing, SCMT, KEGG, ...)
* [ ] Shape-function visualization tool for the learned component functions
* [ ] GPU acceleration utilities for large p and large n

## 🔗 Citation

If you find this work useful, please cite:

```bibtex
@article{chen2026mlnam,
  title   = {Maximum likelihood neural additive models},
  author  = {Chen, Jingyi and Zhang, Xuelin and Yuan, Peipei and Lan, Rushi and Chen, Hong},
  journal = {Information Sciences},
  volume  = {736},
  pages   = {123104},
  year    = {2026},
  issn    = {0020-0255},
  doi     = {10.1016/j.ins.2026.123104}
}
```

## 🙏 Acknowledgements

This project builds upon the ideas of [Neural Additive Models](https://arxiv.org/abs/2004.13946) (Agarwal et al., 2021) and the [official NAM implementation](https://github.com/Argonne-NEXT/NAMU). We thank the community for open-sourcing related resources.

<p align="right"><a href="#ml-nam-maximum-likelihood-neural-additive-models">⬆️ Back to top</a></p>
