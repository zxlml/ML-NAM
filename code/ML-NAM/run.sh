#!/bin/bash
# ML-NAM (PyTorch) — install dependencies and run the simulation experiments.

pip install -r requirements.txt

# Single run: ML-NAM on the Gaussian simulation (p=10, no outliers)
python model/MLNAM_main.py --dataset sim:gaussian:0:10:1280 --loss_type eml --shape_mae

# MSE-based NAM baseline on the same data
python model/MLNAM_main.py --dataset sim:gaussian:0:10:1280 --loss_type mse

# Full Table 3 reproduction (noise x outlier ratio x dimension grid)
python model/run_simulation.py --repeats 5 --dims 10,100,200,400
