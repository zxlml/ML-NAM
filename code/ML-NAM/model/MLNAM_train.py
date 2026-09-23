# coding=utf-8
"""Training loop for ML-NAM (PyTorch).

Implements the estimator of Eq. (9)/(11): the NAM is trained by minimizing
the empirical negative log-likelihood with batch-level KDE (loss_type='eml')
or the least-squares loss (loss_type='mse', NAM baseline), with optional
feature-selection regularization (ML-NAM+).

Training setup follows the paper (Sec. 4, Parameter Selection):
  * Adam optimizer, learning rate 1e-3 with polynomial decay (power = 0.5)
  * mini-batch training (batch size grid {32, 64, 128})
  * early stopping on the validation RMSE/MSE
"""

import copy

import numpy as np
import torch

import graph_builder as gb
import models


def init_output_bias(model, y_train):
    """Initializes the global bias to the scaled mean target.

    This centers the initial residuals around zero so that the KDE in the
    EML loss is well conditioned from the first training step (the density
    of the residuals is non-degenerate), which is important for stable
    convergence of the fixed-bandwidth formulation in Eq. (10)-(11).
    """
    with torch.no_grad():
        model.bias.fill_(float(np.mean(y_train)))


def build_model(cfg, num_inputs):
    if cfg.get('use_dnn', False):
        return models.DNN(dropout=cfg.get('dropout', 0.15))
    return models.NAM(
        num_inputs=num_inputs,
        num_units=cfg.get('num_units', 32),
        num_hidden_layers=cfg.get('num_hidden_layers', 1),
        activation=cfg.get('activation', 'exu'),
        dropout=cfg.get('dropout', 0.0),
        feature_dropout=cfg.get('feature_dropout', 0.0))


def train_model(data, cfg, seed=0):
    """Trains one model and returns the best (val-selected) metrics.

    Args:
      data: dict from data_utils.generate_simulation_data.
      cfg: dict with keys loss_type ('eml'|'mse'), bandwidth, learning_rate,
        batch_size, training_epochs, early_stopping_epochs, num_units,
        num_hidden_layers, activation, dropout, feature_dropout,
        output_regularization, l2_regularization, polynomial_power.
      seed: torch seed.

    Returns:
      dict with train/val/test MSE and MAE (in the scaled target space) and
      the number of epochs trained.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train = torch.as_tensor(data['x_train'], dtype=torch.float32)
    y_train = torch.as_tensor(data['y_train'], dtype=torch.float32)

    model = build_model(cfg, num_inputs=x_train.shape[1])
    init_output_bias(model, data['y_train'])

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get('learning_rate', 1e-3))
    batch_size = min(cfg.get('batch_size', 128), x_train.shape[0])
    epochs = cfg.get('training_epochs', 500)
    patience = cfg.get('early_stopping_epochs', 60)

    best_val = np.inf
    best_state = None
    best_epoch = 0
    n = x_train.shape[0]

    for epoch in range(1, epochs + 1):
        lr = gb.polynomial_decay(epoch, epochs,
                                 cfg.get('learning_rate', 1e-3),
                                 power=cfg.get('polynomial_power', 0.5))
        for group in optimizer.param_groups:
            group['lr'] = lr

        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = x_train[idx], y_train[idx]
            optimizer.zero_grad()
            loss = gb.penalized_loss(
                model, xb, yb,
                base_loss=cfg.get('loss_type', 'eml'),
                bandwidth=cfg.get('bandwidth', 0.05),
                output_regularization=cfg.get('output_regularization', 0.0),
                l2_regularization=cfg.get('l2_regularization', 0.0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg.get('clip_grad_norm', 1.0))
            optimizer.step()

        val_mse, _ = gb.evaluate(model, data['x_val'], data['y_val'])
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    results = {'epochs': best_epoch}
    for split in ('train', 'val', 'test'):
        m, a = gb.evaluate(model, data['x_' + split], data['y_' + split])
        results[split + '_mse'] = m
        results[split + '_mae'] = a
    results['model'] = model
    return results


def shape_function_mae(model, data, grid_size=101):
    """Average MAE between estimated and true component functions (Table 4).

    Only meaningful when p == 10 so that every feature is informative. The
    true component functions are mapped to the scaled target space (they are
    part of f*, which shares the target transformation) and the estimated
    shapes are median-aligned with them to account for the global bias
    absorbed by the model.
    """
    import data_utils
    model.eval()
    grid = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    xg = torch.as_tensor(np.tile(grid, (10, 1)).T, dtype=torch.float32)
    with torch.no_grad():
        outs = model.calc_outputs(xg).cpu().numpy()      # [grid, 10]
    scale = data['y_max'] - data['y_min']
    maes = []
    for j in range(10):
        xx = np.zeros((grid_size, 10), dtype=np.float64)
        xx[:, j] = grid
        true_j = (data_utils.f_star(xx) - data['y_min']) / scale
        offset = np.median(true_j - outs[:, j])
        maes.append(np.mean(np.abs(outs[:, j] + offset - true_j)))
    return float(np.mean(maes))
