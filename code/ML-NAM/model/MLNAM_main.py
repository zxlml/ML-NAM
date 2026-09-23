# coding=utf-8
"""Command-line entry point for training ML-NAM on a simulation dataset.

Examples:
  python MLNAM_main.py --dataset sim:gaussian:0:10:1280 --loss_type eml
  python MLNAM_main.py --dataset sim:mixture:0.1:100:1280 --loss_type mse
  python MLNAM_main.py --dataset sim:gaussian:0:10:1280 --loss_type eml \
      --output_regularization 1e-3   # ML-NAM+
"""

import argparse
import json
import os

import torch

import data_utils
import MLNAM_train


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=str, default='sim:gaussian:0:10:1280',
                        help="Simulation dataset 'sim:<noise>:<r>:<p>:<n>'")
    parser.add_argument('--loss_type', type=str, default='eml',
                        choices=['eml', 'mse'])
    parser.add_argument('--bandwidth', type=float, default=0.05)
    parser.add_argument('--num_units', type=int, default=32)
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    parser.add_argument('--activation', type=str, default='exu',
                        choices=['exu', 'relu'])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--feature_dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--polynomial_power', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--training_epochs', type=int, default=300)
    parser.add_argument('--early_stopping_epochs', type=int, default=30)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--output_regularization', type=float, default=0.0)
    parser.add_argument('--l2_regularization', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shape_mae', action='store_true',
                        help='Report component-function MAE (p=10 only).')
    parser.add_argument('--logdir', type=str, default='../logs')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    data = data_utils.load_simulation_dataset(args.dataset)
    cfg = vars(args)
    results = MLNAM_train.train_model(data, cfg, seed=args.seed)

    print('Dataset: {} | loss: {} | bandwidth: {}'.format(
        args.dataset, args.loss_type, args.bandwidth))
    print('Best epoch: {}'.format(results['epochs']))
    for split in ('train', 'val', 'test'):
        print('{}: MSE {:.4f}  MAE {:.4f}'.format(
            split.capitalize(), results[split + '_mse'], results[split + '_mae']))
    if args.shape_mae and ':' in args.dataset and args.dataset.split(':')[3] == '10':
        print('Shape function MAE: {:.4f}'.format(
            MLNAM_train.shape_function_mae(results['model'], data)))

    os.makedirs(args.logdir, exist_ok=True)
    out = {k: v for k, v in results.items() if k != 'model'}
    out['config'] = {k: v for k, v in vars(args).items()}
    path = os.path.join(
        args.logdir, 'run_{}_{:.0f}pct_seed{}.json'.format(
            args.dataset.replace(':', '_'), 100 * float(args.dataset.split(':')[2]),
            args.seed))
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print('Saved:', path)


if __name__ == '__main__':
    main()
