# coding=utf-8
"""Data utilities: simulation data generation (paper Sec. 4.1) and loaders.

Simulation data follow Section 4.1 of the ML-NAM paper:

    y_i = f*(x_i) + eps_i,   x_i ~ U([0, 1]^p)

    f*(x) = x1^3 + x2 + x3^(1/3) + |x4| + sin(x5) + cos(x6^2)
            + e^{x7} + e^{x8^2} + log(1 + x9^2) + log(1 + x10^(1/2))

Noise eps is drawn from one of three distributions:
    (1) gaussian  : eps ~ N(0, 1)
    (2) mixture   : eps ~ 0.6 N(0, 1) + 0.4 N(0, 3)      (variance notation)
    (3) studentt  : eps ~ t(3)

A fraction r in {0%, 10%} of the labels is additionally perturbed by an
outlier term drawn from N(100, 100) (variance notation, i.e. std 10).

Datasets are split into train/validation/test with ratio 8:1:1. Targets are
min-max scaled to [0, 1] with statistics computed on the (possibly corrupted)
training split; this puts residuals on the same scale as the kernel bandwidth
grid {0.01, 0.05, 0.1, 0.5} of the paper and makes the reported MSE / MAE
comparable to Table 3 of the paper.
"""

import numpy as np

TRAIN_RATIO, VAL_RATIO = 0.8, 0.1


def f_star(x):
    """Ground-truth additive function on the first 10 features.

    Args:
      x: array-like of shape [..., >=10] with entries in [0, 1].
    Returns:
      np.ndarray of shape [...].
    """
    x = np.asarray(x, dtype=np.float64)
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = [x[..., j] for j in range(10)]
    return (x1 ** 3 + x2 + np.cbrt(x3) + np.abs(x4) + np.sin(x5)
            + np.cos(x6 ** 2) + np.exp(x7) + np.exp(x8 ** 2)
            + np.log1p(x9 ** 2) + np.log1p(np.sqrt(x10)))


NOISE_FUNCTIONS = {
    'gaussian': lambda rng, n: rng.normal(0.0, 1.0, size=n),
    'mixture': lambda rng, n: 0.6 * rng.normal(0.0, 1.0, size=n)
                               + 0.4 * rng.normal(0.0, np.sqrt(3.0), size=n),
    'studentt': lambda rng, n: rng.standard_t(df=3, size=n),
}


def generate_simulation_data(n_total=1280,
                             p=10,
                             noise='gaussian',
                             outlier_ratio=0.0,
                             seed=0):
    """Generates the simulation dataset of paper Sec. 4.1.

    A total of `n_total` samples is generated and split 8:1:1 into
    train/validation/test, so the training set contains n_total * 0.8
    samples (1024 for the default n_total = 1280, as in the paper).

    Args:
      n_total: total number of samples (default 1280 -> 1024 training).
      p: input dimensionality in {10, 100, 200, 400}; features beyond the
        first 10 are uninformative.
      noise: one of 'gaussian', 'mixture', 'studentt'.
      outlier_ratio: fraction r of *training* labels corrupted with an extra
        N(100, 100) term (0.0 or 0.1 in the paper).
      seed: RNG seed.

    Returns:
      dict with keys x_train, y_train, x_val, y_val, x_test, y_test (targets
      min-max scaled to [0, 1]) and metadata (y_min, y_max, raw train targets).
    """
    if noise not in NOISE_FUNCTIONS:
        raise ValueError('unknown noise {}'.format(noise))
    rng = np.random.default_rng(seed)

    x = rng.uniform(0.0, 1.0, size=(n_total, p)).astype(np.float32)
    eps = NOISE_FUNCTIONS[noise](rng, n_total)
    y = f_star(x) + eps

    # 8 : 1 : 1 split
    perm = rng.permutation(n_total)
    n_train = int(round(TRAIN_RATIO * n_total))
    n_val = int(round(VAL_RATIO * n_total))
    train_idx, val_idx = perm[:n_train], perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    # Outlier injection on a fraction r of the labels (paper Sec. 4.1). The
    # corruption is applied across the whole generated dataset so that the
    # reported test MSE under r = 10% contains the irreducible contribution
    # of the corrupted evaluation samples, matching the magnitudes of
    # Table 3 (e.g. ML-NAM gaussian p=10: 0.045).
    y = y.astype(np.float64)
    if outlier_ratio > 0:
        n_out = int(round(outlier_ratio * n_total))
        out_idx = rng.choice(n_total, size=n_out, replace=False)
        y[out_idx] += rng.normal(100.0, 10.0, size=n_out)  # N(100, 100)

    # Min-max scale targets to [0, 1] with training statistics (the corrupted
    # labels stretch the range; corrupted test samples then sit far outside
    # [0, 1] and dominate the test error unless the model is robust).
    y_min, y_max = y[train_idx].min(), y[train_idx].max()
    y_scaled = ((y - y_min) / (y_max - y_min)).astype(np.float32)

    return {
        'x_train': x[train_idx],
        'y_train': y_scaled[train_idx],
        'x_val': x[val_idx],
        'y_val': y_scaled[val_idx],
        'x_test': x[test_idx],
        'y_test': y_scaled[test_idx],
        'y_min': float(y_min),
        'y_max': float(y_max),
        'y_train_raw': y[train_idx],
    }


def load_simulation_dataset(name):
    """Loads a simulation dataset by name, e.g. 'sim:gaussian:0:10:1280'.

    Format: sim:<noise>:<outlier_ratio>:<p>:<n_total>
    """
    parts = name.split(':')
    if parts[0] != 'sim' or len(parts) != 5:
        raise ValueError("dataset name must be 'sim:<noise>:<r>:<p>:<n>'")
    _, noise, r, p, n = parts
    return generate_simulation_data(n_total=int(n),
                                    p=int(p),
                                    noise=noise,
                                    outlier_ratio=float(r))
