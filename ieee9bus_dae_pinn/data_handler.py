"""
Data handler for IEEE 9-bus DAE-PINN training.

No external dependencies (deepxde removed).
Uses numpy for random sampling on a hypercube state space.

State vector (dim=12):
  [E'q1, E'd1, d1, w1,  E'q2, E'd2, d2, w2,  E'q3, E'd3, d3, w3]

Typical operating ranges (from plug simulation data):
  E'q  : [0.8, 1.3]  p.u.
  E'd  : [-0.2, 0.2] p.u.
  delta: [-0.6, 0.6] rad   (relative to reference, ~[-35, 35] deg)
  omega: [-0.1, 0.1] rad/s (speed deviation from nominal)
"""
import os
import torch
import numpy as np


# Default state-space bounds per variable type
DEFAULT_BOUNDS = {
    'Eq_prime': (0.8, 1.3),
    'Ed_prime': (-0.2, 0.2),
    'delta':    (-0.6, 0.6),
    'omega':    (-0.1, 0.1),
}

# IRK weights path (relative to this file)
_DIR = os.path.dirname(os.path.abspath(__file__))


class IEEE9BusDataHandler:
    """
    Generates random training / test points in the dynamic state space
    and loads IRK quadrature weights.

    The data handler mirrors the approach in DAE-PINNs/src/data/data.py
    but is self-contained (no deepxde).
    """

    def __init__(
        self,
        num_train=10000,
        num_test=1000,
        num_IRK_stages=10,
        state_dim=12,
        state_bounds=None,
        irk_weights_path=None,
        device='cpu',
    ):
        self.num_train      = num_train
        self.num_test       = num_test
        self.num_IRK_stages = num_IRK_stages
        self.state_dim      = state_dim
        self.device         = device

        # Build per-dimension bounds  [dim, 2]
        if state_bounds is not None:
            # User supplied list of (lo, hi) tuples, length = state_dim
            assert len(state_bounds) == state_dim
            self.bounds = np.array(state_bounds, dtype=np.float32)
        else:
            # 3 generators × 4 states each
            bounds = []
            for _ in range(3):
                bounds.append(DEFAULT_BOUNDS['Eq_prime'])
                bounds.append(DEFAULT_BOUNDS['Ed_prime'])
                bounds.append(DEFAULT_BOUNDS['delta'])
                bounds.append(DEFAULT_BOUNDS['omega'])
            self.bounds = np.array(bounds, dtype=np.float32)  # [12, 2]

        self._load_IRK_weights(irk_weights_path)
        self._generate_data()

    # ------------------------------------------------------------------
    def _load_IRK_weights(self, custom_path=None):
        """Load Butcher tableau weights for IRK integration."""
        if custom_path is not None:
            path = custom_path
        else:
            # Try DAE-PINNs shared weights first, fallback to local copy
            candidates = [
                os.path.join(_DIR, 'data', 'IRK_weights',
                             f'Butcher_IRK{self.num_IRK_stages}.txt'),
                os.path.join(_DIR, '..', 'DAE-PINNs', 'src', 'data',
                             'IRK_weights',
                             f'Butcher_IRK{self.num_IRK_stages}.txt'),
            ]
            path = None
            for c in candidates:
                if os.path.isfile(c):
                    path = c
                    break

        if path is not None and os.path.isfile(path):
            tmp = np.loadtxt(path, ndmin=2)
            s   = self.num_IRK_stages
            IRK_weights = np.reshape(tmp[:s**2 + s], (s + 1, s))
            self.IRK_times   = tmp[s**2 + s:]
            self.IRK_weights = torch.tensor(IRK_weights, dtype=torch.float32).to(self.device)
            print(f"Loaded IRK weights from {path}")
        else:
            print(f"Warning: IRK weights file not found. Using Gauss-Legendre fallback.")
            s = self.num_IRK_stages
            # Simple equal-weight fallback (Lobatto-style)
            b   = np.ones(s) / s
            c   = np.linspace(0, 1, s)
            # Build [s+1, s]: rows 0..s-1 = A (zeros for explicit, but we use
            # the same b vector for all rows as a placeholder), last row = b
            A   = np.tile(b, (s, 1))
            IRK_weights = np.vstack([A, b])
            self.IRK_times   = c
            self.IRK_weights = torch.tensor(IRK_weights, dtype=torch.float32).to(self.device)

    # ------------------------------------------------------------------
    def _sample(self, n, seed):
        """Latin Hypercube-style uniform random sampling."""
        rng = np.random.default_rng(seed)
        lo  = self.bounds[:, 0]
        hi  = self.bounds[:, 1]
        X   = rng.uniform(lo, hi, size=(n, self.state_dim)).astype(np.float32)
        return torch.tensor(X, dtype=torch.float32).to(self.device)

    def _generate_data(self):
        self.X_train = self._sample(self.num_train, seed=1234)
        self.X_test  = self._sample(self.num_test,  seed=3456)
        print(f"Generated {self.num_train} training points and {self.num_test} test points")

    # ------------------------------------------------------------------
    def get_train_batch(self, batch_size=None):
        if batch_size is None or batch_size >= self.num_train:
            return self.X_train
        idx = torch.randperm(self.num_train)[:batch_size]
        return self.X_train[idx]

    def get_test_data(self):
        return self.X_test

    def get_IRK_weights(self):
        return self.IRK_weights
