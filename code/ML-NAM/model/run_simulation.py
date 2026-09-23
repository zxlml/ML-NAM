# coding=utf-8
"""Reproduces the simulation experiments of Table 3 in the ML-NAM paper.

For every configuration (noise in {gaussian, mixture, studentt} x outlier
ratio r in {0%, 10%} x dimensionality p in {10, 100, 200, 400}) the script
trains:
    * NAM      : MSE loss (least squares baseline)
    * ML-NAM   : maximum-likelihood loss with KDE (eml)
    * ML-NAM+  : ML-NAM + feature-selection regularization
and reports average MSE / MAE (std over repeats) in the scaled target space,
to be compared with Table 3 of the paper.

Usage:
  python run_simulation.py --repeats 5 --dims 10,100,200,400
  python run_simulation.py --quick          # small sanity run (p=10 only)
"""

import argparse
import csv
import os
import time

import data_utils
import MLNAM_train

# Reference values from Table 3 of the paper (ML-NAM test MSE).
PAPER_REF = {
    ('gaussian', 0.0, 10): 0.014, ('gaussian', 0.0, 100): 0.110,
    ('gaussian', 0.0, 200): 0.125, ('gaussian', 0.0, 400): 0.167,
    ('mixture', 0.0, 10): 0.035, ('mixture', 0.0, 100): 0.156,
    ('mixture', 0.0, 200): 0.186, ('mixture', 0.0, 400): 0.353,
    ('studentt', 0.0, 10): 0.032, ('studentt', 0.0, 100): 0.142,
    ('studentt', 0.0, 200): 0.157, ('studentt', 0.0, 400): 0.191,
    ('gaussian', 0.1, 10): 0.045, ('gaussian', 0.1, 100): 0.120,
    ('gaussian', 0.1, 200): 0.147, ('gaussian', 0.1, 400): 0.235,
    ('mixture', 0.1, 10): 0.069, ('mixture', 0.1, 100): 0.208,
    ('mixture', 0.1, 200): 0.315, ('mixture', 0.1, 400): 0.438,
    ('studentt', 0.1, 10): 0.048, ('studentt', 0.1, 100): 0.155,
    ('studentt', 0.1, 200): 0.172, ('studentt', 0.1, 400): 0.248,
}

BASE_CFG = dict(
    num_units=32,
    num_hidden_layers=1,
    activation='exu',
    dropout=0.1,
    feature_dropout=0.1,
    learning_rate=1e-3,
    polynomial_power=0.5,
    batch_size=128,
    training_epochs=300,
    early_stopping_epochs=30,
    clip_grad_norm=1.0,
    bandwidth=0.05,
    output_regularization=0.0,
    l2_regularization=0.0,
)

MODEL_VARIANTS = {
    'NAM': dict(BASE_CFG, loss_type='mse'),
    'ML-NAM': dict(BASE_CFG, loss_type='eml'),
    'ML-NAM+': dict(BASE_CFG, loss_type='eml', output_regularization=1e-3),
}


def run_config(noise, outlier_ratio, dim, repeats, n_total, epochs=None):
    rows = []
    for rep in range(repeats):
        data = data_utils.generate_simulation_data(
            n_total=n_total, p=dim, noise=noise,
            outlier_ratio=outlier_ratio, seed=1000 + rep)
        for model_name, cfg in MODEL_VARIANTS.items():
            cfg = dict(cfg)
            if epochs is not None:
                cfg['training_epochs'] = epochs
            t0 = time.time()
            res = MLNAM_train.train_model(data, cfg, seed=rep)
            rows.append({
                'noise': noise, 'outlier_ratio': outlier_ratio, 'p': dim,
                'model': model_name, 'repeat': rep,
                'train_mse': res['train_mse'], 'train_mae': res['train_mae'],
                'test_mse': res['test_mse'], 'test_mae': res['test_mae'],
                'best_epoch': res['epochs'],
                'seconds': round(time.time() - t0, 1),
            })
            print('[{} r={:.0f}% p={}] {} rep{}: test MSE {:.4f} MAE {:.4f} '
                  '(epoch {}, {:.1f}s)'.format(
                      noise, 100 * outlier_ratio, dim, model_name, rep,
                      res['test_mse'], res['test_mae'], res['epochs'],
                      time.time() - t0), flush=True)
    return rows


def summarize(rows):
    import numpy as np
    print('\n=== Summary (test MSE / MAE, mean +- std) vs paper reference ===')
    print('{:<10} {:<6} {:<5} {:<9} {:<16} {:<16} {:<10}'.format(
        'Noise', 'r', 'p', 'Model', 'MSE', 'MAE', 'Paper MSE'))
    agg = {}
    for r in rows:
        agg.setdefault((r['noise'], r['outlier_ratio'], r['p'], r['model']),
                       []).append((r['test_mse'], r['test_mae']))
    for key in sorted(agg):
        vals = agg[key]
        mses = [v[0] for v in vals]
        maes = [v[1] for v in vals]
        ref = PAPER_REF.get((key[0], key[1], key[2]), float('nan'))
        if key[3] == 'ML-NAM':
            print('{:<10} {:<6} {:<5} {:<9} {:.3f}+-{:.3f}   {:.3f}+-{:.3f}   '
                  '{:<10}'.format(
                      key[0], key[1], key[2], key[3],
                      np.mean(mses), np.std(mses), np.mean(maes), np.std(maes),
                      ref))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--dims', type=str, default='10,100,200,400')
    parser.add_argument('--noises', type=str, default='gaussian,mixture,studentt')
    parser.add_argument('--outlier_ratios', type=str, default='0.0,0.1')
    parser.add_argument('--n_total', type=int, default=1280,
                        help='1280 -> 1024 training samples as in the paper')
    parser.add_argument('--training_epochs', type=int, default=None)
    parser.add_argument('--quick', action='store_true',
                        help='Sanity run: p=10 only, 2 repeats, 200 epochs')
    parser.add_argument('--out_csv', type=str,
                        default='../logs/simulation_results.csv')
    args = parser.parse_args()

    if args.quick:
        args.dims, args.repeats = '10', 2
        args.training_epochs = args.training_epochs or 200

    dims = [int(d) for d in args.dims.split(',')]
    noises = args.noises.split(',')
    ratios = [float(r) for r in args.outlier_ratios.split(',')]

    all_rows = []
    for noise in noises:
        for ratio in ratios:
            for dim in dims:
                all_rows += run_config(noise, ratio, dim, args.repeats,
                                       args.n_total, args.training_epochs)

    summarize(all_rows)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print('\nSaved raw results to', args.out_csv)


if __name__ == '__main__':
    main()
