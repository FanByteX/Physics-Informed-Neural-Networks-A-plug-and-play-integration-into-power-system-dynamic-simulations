"""
Trainer for IEEE 9-bus DAE-PINN.

Save format is compatible with plug/src/tds_dae_rk_schemes.py:
  checkpoint = {
      'state_dict'         : model.state_dict(),
      'architecture'       : (_, num_neurons, num_layers, inputs, outputs),
      'machine_parameters' : (D, Pg, H, Xd_p, Rs)   -- per machine tuple
      'range_norm'         : (norm_range, lb_range)  -- dummy / actual norm
      'voltage_stats'      : (voltage_limits, ...)
      'theta_stats'        : (theta_limits, ...)
      'init_state'         : (Eq0, Ed0, delta0, omega0)
      'model_config'       : {full args dict}
      'epoch'              : int
      'loss'               : float
  }
"""
import os
import time
import torch
import numpy as np
from tqdm import tqdm

from models import IEEE9Bus_PINN
from physics import IEEE9BusPhysics, dotdict, compute_total_loss
from data_handler import IEEE9BusDataHandler


class Trainer:
    def __init__(
        self,
        config_dynamic_path='../config_files/config_machines_dynamic.yaml',
        config_static_path='../config_files/config_machines_static.yaml',
        Y_admittance_path='../config_files/network_admittance.pt',
        log_dir='./logs',
        device='cuda',
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {self.device}")

        # Create a timestamped run directory
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(log_dir, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        print(f"Run directory: {self.run_dir}")

        self.physics      = IEEE9BusPhysics(config_dynamic_path, config_static_path, Y_admittance_path)
        self.data_handler = None
        self.model        = None
        self.optimizer    = None
        self.scheduler    = None
        self.loss_history = []
        self.best_loss    = float('inf')
        self.best_epoch   = 0

        # store config paths for saving
        self._cfg_dyn  = config_dynamic_path
        self._cfg_sta  = config_static_path
        self._cfg_Yadm = Y_admittance_path

    # ------------------------------------------------------------------
    def setup_model(
        self,
        num_IRK_stages=10,
        dyn_layer_size=None,
        alg_layer_size=None,
        activation='tanh',
        stacked=True,
    ):
        if dyn_layer_size is None:
            dyn_layer_size = [12, 64, 64, 64, num_IRK_stages + 1]
        if alg_layer_size is None:
            alg_layer_size = [12, 64, 64, 18 * (num_IRK_stages + 1)]

        dynamic = dotdict({
            'num_IRK_stages': num_IRK_stages,
            'layer_size':     dyn_layer_size,
            'activation':     activation,
            'initializer':    'Glorot normal',
            'dropout_rate':   None,
        })
        algebraic = dotdict({
            'num_IRK_stages': num_IRK_stages,
            'layer_size':     alg_layer_size,
            'activation':     activation,
            'initializer':    'Glorot normal',
            'dropout_rate':   None,
        })

        self.model = IEEE9Bus_PINN(
            dynamic, algebraic, stacked=stacked,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model created: {n_params:,} parameters  (stacked={stacked})")

        # store for checkpoint
        self._num_IRK_stages = num_IRK_stages
        self._dyn_layer_size = dyn_layer_size
        self._alg_layer_size = alg_layer_size
        self._activation     = activation
        self._stacked        = stacked

    # ------------------------------------------------------------------
    def setup_data(self, num_train=10000, num_test=1000, num_IRK_stages=10):
        self.data_handler = IEEE9BusDataHandler(
            num_train=num_train,
            num_test=num_test,
            num_IRK_stages=num_IRK_stages,
            state_dim=12,
            device=self.device,
        )

    # ------------------------------------------------------------------
    def setup_optimizer(self, lr=1e-3, scheduler_type=None, patience=1000, factor=0.5):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=patience, factor=factor, verbose=True)
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=patience, gamma=factor)
        else:
            self.scheduler = None

    # ------------------------------------------------------------------
    def resume(self, checkpoint_path):
        """Load weights from an existing checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.best_loss = ckpt.get('loss', float('inf'))
        print(f"Resumed from {checkpoint_path}  (loss={self.best_loss:.4e})")

    # ------------------------------------------------------------------
    def train(
        self,
        epochs=10000,
        batch_size=None,
        h=0.04,
        loss_weights=None,
        test_every=100,
        save_every=1000,
        model_name='ieee9bus_pinn',
        use_tqdm=True,
    ):
        if loss_weights is None:
            loss_weights = [1.0, 1.0]

        print(f"\nStarting training for {epochs} epochs")
        print(f"Loss weights: dynamic={loss_weights[0]}, algebraic={loss_weights[1]}")

        IRK_weights = self.data_handler.get_IRK_weights()
        h_t = torch.tensor([h], dtype=torch.float32).to(self.device)

        t0 = time.time()
        iterator = tqdm(range(epochs)) if use_tqdm else range(epochs)

        for epoch in iterator:
            self.model.train()
            X = self.data_handler.get_train_batch(batch_size)

            self.optimizer.zero_grad()
            f_res, g_res = self.physics.compute_IRK_residuals(
                self.model, X, h_t, IRK_weights, str(self.device))
            loss, loss_dict = compute_total_loss(f_res, g_res, loss_weights)
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss)
                else:
                    self.scheduler.step()

            self.loss_history.append(loss_dict)

            if (epoch + 1) % test_every == 0:
                test_loss = self._test(epoch, h_t, IRK_weights, loss_weights)
                if test_loss < self.best_loss:
                    self.best_loss  = test_loss
                    self.best_epoch = epoch + 1
                    self._save(model_name, epoch, is_best=True)

            if (epoch + 1) % save_every == 0:
                self._save(model_name, epoch, is_best=False)

        elapsed = time.time() - t0
        print(f"\nTraining done in {elapsed:.1f}s  |  Best loss: {self.best_loss:.4e}  @ epoch {self.best_epoch}")
        self._save_loss_history()

    # ------------------------------------------------------------------
    def _test(self, epoch, h, IRK_weights, loss_weights):
        self.model.eval()
        with torch.no_grad():
            X = self.data_handler.get_test_data()
            f_res, g_res = self.physics.compute_IRK_residuals(
                self.model, X, h, IRK_weights, str(self.device))
            loss, ld = compute_total_loss(f_res, g_res, loss_weights)
        lv = loss.item()
        print(f"\nEpoch {epoch+1:>6d} | test_total={lv:.4e}"
              f"  dyn={ld['loss_dyn']:.4e}  alg={ld['loss_alg']:.4e}")
        self.model.train()
        return lv

    # ------------------------------------------------------------------
    def _build_checkpoint(self, epoch, loss):
        """
        Build a checkpoint dict compatible with plug inference code.
        plug expects keys: state_dict, architecture, machine_parameters,
        range_norm, voltage_stats, theta_stats, init_state
        """
        ph = self.physics

        # machine_parameters per generator: (D, Pg, H, Xd_p, Rs)
        machine_params = []
        for i in range(3):
            machine_params.append((
                ph.D[i].item(),
                ph.Pg[i].item(),
                ph.H[i].item(),
                ph.Xd_p[i].item(),
                ph.Rs[i].item() if hasattr(ph, 'Rs') else 0.0,
            ))

        # architecture tuple: (_, num_neurons, num_layers, inputs, outputs)
        dls = self._dyn_layer_size
        arch = (None, dls[1], len(dls) - 2, dls[0], dls[-1])

        # norm range: use state-space bounds from data handler
        if self.data_handler is not None:
            lo  = self.data_handler.bounds[:, 0].tolist()
            hi  = self.data_handler.bounds[:, 1].tolist()
        else:
            lo = [-1.0] * 12
            hi = [1.0]  * 12
        norm_range = list(zip(lo, hi))
        lb_range   = lo

        # operating limits from state-space bounds
        voltage_limits = (0.8, 1.3)      # E'q approx range
        theta_limits   = (-0.6, 0.6)     # delta range
        delta_limits   = [lo[2], hi[2]]  # gen-1 delta
        omega_limits   = [lo[3], hi[3]]  # gen-1 omega

        ckpt = {
            'state_dict':         self.model.state_dict(),
            'architecture':       arch,
            'machine_parameters': machine_params,
            'range_norm':         (norm_range, lb_range),
            'voltage_stats':      (voltage_limits,),
            'theta_stats':        (theta_limits,),
            'init_state':         (lo[0], lo[1], delta_limits, omega_limits),
            'model_config': {
                'stacked':        self._stacked,
                'num_IRK_stages': self._num_IRK_stages,
                'dyn_layer_size': self._dyn_layer_size,
                'alg_layer_size': self._alg_layer_size,
                'activation':     self._activation,
            },
            'epoch': epoch,
            'loss':  loss,
        }
        return ckpt

    def _save(self, model_name, epoch, is_best=False):
        loss = self.loss_history[-1]['loss_total'] if self.loss_history else float('inf')
        ckpt = self._build_checkpoint(epoch, loss)

        if is_best:
            path = os.path.join(self.run_dir, f'{model_name}_best.pth')
        else:
            path = os.path.join(self.run_dir, f'{model_name}_epoch{epoch}.pth')

        torch.save(ckpt, path)
        if is_best:
            print(f"  → Best model saved: {path}")

    def _save_loss_history(self):
        if not self.loss_history:
            return
        path = os.path.join(self.run_dir, 'loss_history.npz')
        total = np.array([d['loss_total'] for d in self.loss_history])
        dyn   = np.array([d['loss_dyn']   for d in self.loss_history])
        alg   = np.array([d['loss_alg']   for d in self.loss_history])
        np.savez(path, loss_total=total, loss_dyn=dyn, loss_alg=alg)
        print(f"Loss history saved: {path}")
