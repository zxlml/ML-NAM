# coding=utf-8
"""ML-NAM model definitions (PyTorch).

Rewritten from the original TensorFlow 1.x implementation so that the project
runs on modern Python (>=3.10). The architecture follows Eq. (4)-(6) of the
ML-NAM paper:

    f(x) = sum_{j=1..p} f_j(x_j)

where each component function f_j is a small neural network applied to a
single input feature (vectorized over features for efficiency). The first
hidden layer uses either ExU units (default, as in Agarwal et al. 2021) or
threshold ReLU units phi_v(a) = max(a - v, 0) as written in Eq. (5).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NAM(nn.Module):
    """Neural Additive Model, vectorized over the p feature subnetworks.

    All feature subnetworks share the same shape but have independent
    weights, stored as tensors with a leading feature dimension so that a
    single einsum computes all p subnetworks in parallel.
    """

    def __init__(self,
                 num_inputs,
                 num_units=32,
                 num_hidden_layers=1,
                 activation='exu',
                 dropout=0.0,
                 feature_dropout=0.0):
        """
        Args:
          num_inputs: number of input features p.
          num_units: hidden units per layer of each subnetwork (paper grid
            {16, 32, 64}).
          num_hidden_layers: number of hidden layers per subnetwork (paper
            grid {1, 2, 3}).
          activation: 'exu' or 'relu' for the first hidden layer.
          dropout: dropout rate on hidden activations of each subnetwork.
          feature_dropout: dropout rate on the per-feature outputs.
        """
        super().__init__()
        self._num_inputs = int(num_inputs)
        self._num_units = int(num_units)
        self._num_hidden_layers = int(num_hidden_layers)
        self._activation = activation
        self._dropout = float(dropout)
        self._feature_dropout = float(feature_dropout)
        p, U = self._num_inputs, self._num_units

        if activation == 'exu':
            # ExU: exp(beta) * (x - c), beta ~ TruncN(4, 0.5) as in the
            # original NAM implementation.
            self.beta = nn.Parameter(torch.empty(p, U))
            nn.init.trunc_normal_(self.beta, mean=4.0, std=0.5, a=1.0, b=7.0)
        elif activation == 'relu':
            # Threshold ReLU phi_v(a) = relu(w * (x - c)) as in Eq. (5).
            self.w0 = nn.Parameter(torch.empty(p, U))
            nn.init.normal_(self.w0, mean=1.0, std=0.5)
        else:
            raise ValueError('{} is not a valid activation'.format(activation))
        self.c0 = nn.Parameter(torch.empty(p, U))
        nn.init.trunc_normal_(self.c0, mean=0.0, std=0.5, a=-1.5, b=1.5)

        # Additional hidden layers (Dense + ReLU), one weight tensor per layer.
        self.hidden_weights = nn.ParameterList()
        self.hidden_biases = nn.ParameterList()
        for _ in range(self._num_hidden_layers - 1):
            W = nn.Parameter(torch.empty(p, U, U))
            nn.init.xavier_uniform_(W)
            self.hidden_weights.append(W)
            self.hidden_biases.append(nn.Parameter(torch.zeros(p, U)))

        # Output projection of each subnetwork (no bias, as in FeatureNN).
        # Zero-initialized so that the initial prediction equals the global
        # bias (mean of the scaled targets); this keeps the initial residual
        # spread small and dense, which is required for the fixed-bandwidth
        # KDE in the EML loss to be well conditioned (see graph_builder).
        self.v_out = nn.Parameter(torch.zeros(p, U))

        # Global bias; initialized to zero (set to mean(y) by the trainer for
        # faster convergence, see MLNAM_train.init_output_bias).
        self.bias = nn.Parameter(torch.zeros(1))

    def _first_layer(self, x, training):
        if self._activation == 'exu':
            h = torch.exp(self.beta) * (x.unsqueeze(-1) - self.c0)
            # relu_n: clip ExU outputs to [0, 1] as in the original NAM code.
            h = torch.clamp(torch.relu(h), max=1.0)
        else:
            h = torch.relu(self.w0 * x.unsqueeze(-1) - self.c0)
        if training and self._dropout > 0:
            h = F.dropout(h, p=self._dropout, training=True)
        return h

    def calc_outputs(self, x):
        """Per-feature outputs f_j(x_j), shape [batch, p] (no dropout)."""
        h = self._first_layer(x, training=False)
        for W, b in zip(self.hidden_weights, self.hidden_biases):
            h = torch.relu(torch.einsum('bpu,puv->bpv', h, W) + b)
        return torch.einsum('bpu,pu->bp', h, self.v_out)

    def forward(self, x, training=True):
        h = self._first_layer(x, training)
        for W, b in zip(self.hidden_weights, self.hidden_biases):
            h = torch.relu(torch.einsum('bpu,puv->bpv', h, W) + b)
            if training and self._dropout > 0:
                h = F.dropout(h, p=self._dropout, training=True)
        out = torch.einsum('bpu,pu->bp', h, self.v_out)   # [B, p]
        if training and self._feature_dropout > 0:
            out = F.dropout(out, p=self._feature_dropout, training=True)
        return out.sum(dim=-1) + self.bias


class DNN(nn.Module):
    """Deep Neural Network baseline (10 hidden layers, 100 units, ReLU)."""

    def __init__(self, dropout=0.15):
        super().__init__()
        self._dropout = float(dropout)
        self.hidden = nn.ModuleList()
        self.out = None

    def forward(self, x, training=True):
        if self.out is None:
            d = x.shape[-1]
            for _ in range(10):
                lin = nn.Linear(d, 100)
                nn.init.kaiming_normal_(lin.weight)
                self.hidden.append(lin)
                d = 100
            self.out = nn.Linear(d, 1)
            nn.init.kaiming_normal_(self.out.weight)
            self.out.to(x.device)
            self.hidden.to(x.device)
        for lin in self.hidden:
            x = torch.relu(lin(x))
            if training and self._dropout > 0:
                x = F.dropout(x, p=self._dropout, training=True)
        return self.out(x).squeeze(-1)
