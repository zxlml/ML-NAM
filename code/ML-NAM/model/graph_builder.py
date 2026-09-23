# coding=utf-8
"""Loss functions, metrics and schedules for ML-NAM (PyTorch).

The core of ML-NAM is the maximum-likelihood (EML) loss of Eq. (10)-(11) in
the paper: the residual density h is estimated with kernel density estimation
(KDE) using a *fixed* bandwidth sigma,

    K_sigma(u, u')  = exp( -||u - u'||^2 / (2 sigma^2) )              (10)
    h_hat(u)        = (1/n) sum_i K_sigma(y_i - f(x_i), u)
    L_n(f)          = (1/n) sum_i -log h_hat(y_i - f(x_i))            (11)

Following Remark 1 of the paper, the KDE is computed within each mini-batch
to avoid the O(n^2) cost over the full training set.

The original repo used a k-nearest-neighbour *variable* bandwidth and
sub-sampled the residual reference set, which deviates from the paper's
formulation; both are removed here. To keep the fixed-bandwidth KDE well
conditioned, targets are min-max scaled to [0, 1] (see data_utils) so that
residuals live on the same scale as the bandwidth grid {0.01, 0.05, 0.1, 0.5}
used in the paper, and the model bias is initialized to the (scaled) mean of
the training targets so residuals start out concentrated near zero.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


def eml_loss(pred, target, bandwidth, leave_one_out=True, eps=1e-10):
    """Maximum-likelihood loss with batch-level KDE, Eq. (10)-(11).

    Args:
      pred: model predictions f(x_i), shape [B].
      target: targets y_i, shape [B].
      bandwidth: fixed kernel bandwidth sigma.
      leave_one_out: exclude the self-term K(r_i, r_i) = 1 from the density
        estimate of each residual. With the self-term included, every
        residual is isolated on the plateau -log(1/n) with zero gradient
        once pairwise distances exceed ~2 sigma, so training can drift
        without any restoring force (the likelihood with self-inclusion is
        maximized by a degenerate delta-density). Leave-one-out KDE is the
        standard formulation for maximum-likelihood density estimation
        (Duin, 1976) and keeps the loss well posed: isolated residuals get
        a large loss and a restoring gradient, which also underlies the
        robustness of ML-NAM to outliers.
    """
    r = (target - pred).reshape(-1)
    d2 = (r.unsqueeze(0) - r.unsqueeze(1)).pow(2)                 # [B, B]
    k = torch.exp(-d2 / (2.0 * float(bandwidth) ** 2))
    if leave_one_out:
        eye = torch.eye(k.shape[0], dtype=torch.bool, device=k.device)
        k = k.masked_fill(eye, 0.0)                               # remove self
        h = k.sum(dim=1) / (k.shape[1] - 1)
    else:
        h = k.mean(dim=1)                                         # [B]
    return -torch.log(h.clamp_min(eps)).mean()


def mse_loss(pred, target):
    """Least-squares loss (Eq. 2), used by the MSE-based NAM baseline."""
    return F.mse_loss(pred.reshape(-1), target.reshape(-1))


def feature_output_regularization(model, inputs):
    """Penalizes the L2 norm of each feature net output (for ML-NAM+)."""
    per_feature_outputs = model.calc_outputs(inputs)               # [B, p]
    return per_feature_outputs.pow(2).mean()


def weight_decay(model, num_networks=1):
    """L2 penalty over all trainable weights (tf.nn.l2_loss convention)."""
    l2 = 0.0
    for param in model.parameters():
        l2 = l2 + param.pow(2).sum()
    return l2 / (2.0 * num_networks)


def penalized_loss(model,
                   inputs,
                   targets,
                   base_loss,
                   bandwidth=None,
                   output_regularization=0.0,
                   l2_regularization=0.0):
    """Base loss plus ML-NAM+ feature-selection and L2 regularization."""
    pred = model(inputs, training=True)
    if base_loss == 'eml':
        loss = eml_loss(pred, targets, bandwidth)
    else:
        loss = mse_loss(pred, targets)
    if output_regularization > 0:
        loss = loss + output_regularization * feature_output_regularization(
            model, inputs)
    if l2_regularization > 0:
        # One "network" per feature for NAM-style models, 1 for the DNN.
        num_networks = 1 if hasattr(model, 'hidden') else \
            getattr(model, '_num_inputs', 1)
        loss = loss + l2_regularization * weight_decay(
            model, num_networks=num_networks)
    return loss


def polynomial_decay(epoch, total_epochs, base_lr, power=0.5, min_lr=1e-6):
    """Polynomial learning-rate decay (power = 0.5), as in the paper.

    lr(epoch) = base_lr * (1 - epoch/total_epochs)^power
    """
    progress = min(1.0, float(epoch) / float(max(1, total_epochs)))
    return max(min_lr, base_lr * math.pow(1.0 - progress, power))


# ----------------------------------------------------------------------------
# Metrics (computed in the scaled target space, matching Table 3 of the paper)
# ----------------------------------------------------------------------------

def mse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate(model, x, y, batch_size=4096):
    """Returns (mse, mae) of the model on a dataset (eval mode, no grad)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32)
            preds.append(model(xb, training=False).cpu().numpy())
    preds = np.concatenate(preds)
    return mse(y, preds), mae(y, preds)
