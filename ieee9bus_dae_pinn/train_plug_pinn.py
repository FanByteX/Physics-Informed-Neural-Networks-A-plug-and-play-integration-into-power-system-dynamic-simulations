"""
Train a plug-compatible single-machine PINN for machine 1.
Reproduces the training logic from the original paper:

  Ventura et al. "Physics-Informed Neural Networks: a Plug and Play Integration
  into Power System Dynamic Simulations". 2024.

Network (PINN as integrator with hard constraint):
  Output is derivatives, but NOT d(delta)/dt directly!

  The key insight from plug's Newton residual:
    res_2 = delta1 - h*d_pinn - omicron_0 - theta_pend*h - Theta0 = 0
    => delta1 = h*d_pinn + omicron_0 + theta_pend*h + Theta0

  Substituting omicron_0 = delta0 - Theta0 and theta_pend = (Theta1 - Theta0)/h:
    delta1 = h*d_pinn + delta0 + Theta1 - Theta0

  Therefore PINN output1 = d(omicron)/dt = d(delta)/dt - d(Theta)/dt
                         = omega * 2πf - theta_pend
  And        PINN output2 = d(omega)/dt

  Hard constraint:
    omicron_1 = omicron_0 + h * d(omicron)/dt
    omega_1 = omega_0 + h * d(omega)/dt

  This formulation avoids needing omicron_1 (which requires delta1) during inference.

Input (6-dim):  [Vm_0, Vm_1, theta_pend, omicron_0, omega_0, h]
  Vm_0, Vm_1    : terminal voltage at t=n and t=n+1
  theta_pend    : dθ/dt = (θ_1 - θ_0) / h
  omicron_0     : δ_0 - θ_0  (angle in machine frame, t=n)
  omega_0       : rotor speed deviation at t=n
  h             : time step size (randomly sampled during training)

Output (2-dim): [d(omicron)/dt, d(omega)/dt]
  d(omicron)/dt = d(delta)/dt - d(Theta)/dt  (NOT directly d(delta)/dt!)
  d(omega)/dt   : rotor acceleration

Loss = L_data + alpha * L_physics
  L_data   : MSE( NN_output, true_derivatives ) from TDS simulation
  L_physics: MSE( d(NN)/dτ, f(x_τ, y_τ) )  AutoDiff residual at τ∈[0,h]
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn

# Use float64 (double precision) for better numerical accuracy, matching original plug model
torch.set_default_dtype(torch.float64)
from tqdm import tqdm

# ── Use local src modules (independent of plug) ───────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)   # project root: /home/PINNs-Plug-n-Play-Integration
sys.path.insert(0, os.path.join(_HERE, 'src'))

import yaml


# =============================================================================
# Network architectures (matching plug's model structures)
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual block for FCN_RESNET."""
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        identity = x
        out = torch.tanh(self.fc1(x))
        out = self.fc2(out)
        out = out + identity
        out = torch.tanh(out)
        return out


class FCN_RESNET(nn.Module):
    """
    ResNet architecture matching plug's final_models weight structure:
      fcs.0.*  (input layer: Linear + Tanh)
      fch.k.0.* (ResidualBlock layers, k=0..N_LAYERS-2)
      fce.*    (output layer)

    Weight keys saved by plug:
      net.fcs.0.weight, net.fcs.0.bias
      net.fch.0.0.fc1.weight, net.fch.0.0.fc1.bias, net.fch.0.0.fc2.weight, net.fch.0.0.fc2.bias
      net.fce.weight, net.fce.bias
    """

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, range_states, lb_states, device='cpu'):
        super().__init__()
        self.device = device

        # store normalization as buffers
        self.register_buffer('range_states', torch.tensor(range_states, dtype=torch.float64))
        self.register_buffer('lb_states',    torch.tensor(lb_states,    dtype=torch.float64))

        # fcs: input layer (Linear + Tanh)
        self.fcs = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),   # index 0
            nn.Tanh()
        )

        # fch: ResidualBlocks
        hidden_blocks = []
        for _ in range(N_LAYERS - 1):
            hidden_blocks.append(nn.Sequential(ResidualBlock(N_HIDDEN)))
        self.fch = nn.Sequential(*hidden_blocks)

        # output layer
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

        # initialise
        nn.init.xavier_normal_(self.fcs[0].weight)
        for blk in self.fch:
            nn.init.xavier_normal_(blk[0].fc1.weight)
            nn.init.xavier_normal_(blk[0].fc2.weight)
        nn.init.xavier_normal_(self.fce.weight)

    def forward(self, x):
        """Returns derivative [d_omicron/dt, d_omega/dt]."""
        two = torch.tensor(2.0, dtype=x.dtype, device=x.device)
        one = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        x = two * (x - self.lb_states.to(x.dtype)) / self.range_states.to(x.dtype) - one

        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


class FCN_simple(nn.Module):
    """
    Simple fully-connected network (alternative to ResNet).
    Weight keys:
      fcs.1.*  (input layer)
      fch.k.0.* (hidden layers, k=0..N_LAYERS-2)
      fce.*    (output layer)

    Normalization in forward(): x_norm = 2*(x - lb) / range - 1
    """

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, range_states, lb_states, device='cpu'):
        super().__init__()
        self.device = device

        # store normalization as buffers (will move with .to(device))
        self.register_buffer('range_states', torch.tensor(range_states, dtype=torch.float32))
        self.register_buffer('lb_states',    torch.tensor(lb_states,    dtype=torch.float32))

        # fcs: Sequential with index 1 = Linear (to match saved keys)
        self.fcs = nn.Sequential(
            nn.Identity(),                    # index 0 (placeholder)
            nn.Linear(N_INPUT, N_HIDDEN),     # index 1  →  fcs.1.*
        )
        # fch: Sequential of hidden layers, each a Sequential with index 0 = Linear
        hidden = []
        for _ in range(N_LAYERS - 1):
            hidden.append(nn.Sequential(
                nn.Linear(N_HIDDEN, N_HIDDEN),  # index 0  →  fch.k.0.*
            ))
        self.fch = nn.Sequential(*hidden)

        # output
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

        # initialise
        nn.init.xavier_normal_(self.fcs[1].weight)
        for blk in self.fch:
            nn.init.xavier_normal_(blk[0].weight)
        nn.init.xavier_normal_(self.fce.weight)

    def forward(self, x):
        """Returns derivative [d_omicron/dt, d_omega/dt]. Hard constraint applied externally."""
        two  = torch.tensor(2.0, dtype=x.dtype, device=x.device)
        one  = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        x = two * (x - self.lb_states.to(x.dtype)) / self.range_states.to(x.dtype) - one

        x = torch.tanh(self.fcs[1](x))
        for blk in self.fch:
            x = torch.tanh(blk[0](x))
        x = self.fce(x)
        return x


# =============================================================================
# Data generation using plug's TDS_simulation
# =============================================================================

def generate_training_data(machine_idx, h, n_traj, n_steps, config_dir, device_str='cpu'):
    """
    Run TDS simulations without PINN boost to collect supervised pairs.

    For each trajectory:
      x_0 = state at step t
      x_1 = state at step t+1  (ground truth from Newton solver)

    From x_0 and x_1 we construct the 6-dim PINN input and 2-dim label.

    machine_idx : 1-based (1, 2, or 3)
    """
    import yaml
    from tds_dae_rk_schemes import TDS_simulation

    dyn_cfg  = os.path.join(config_dir, 'config_machines_dynamic.yaml')
    sta_cfg  = os.path.join(config_dir, 'config_machines_static.yaml')
    Yadm_pt  = os.path.join(config_dir, 'network_admittance.pt')

    with open(dyn_cfg, 'r') as f:
        dyn = yaml.safe_load(f)
    with open(sta_cfg, 'r') as f:
        sta = yaml.safe_load(f)

    freq     = dyn['freq']
    H_vec    = list(dyn['inertia_H'].values())
    Rs_vec   = list(dyn['Rs'].values())
    Xdp_vec  = list(dyn['Xd_prime'].values())
    pg_pf    = list(dyn['Pg_setpoints'].values())
    dampings = list(dyn['Damping_D'].values())
    Yadm     = torch.load(Yadm_pt)

    volt_mag = list(sta['Voltage_magnitude'].values())
    volt_ang = list(sta['Voltage_angle'].values())
    volt_complex = np.array(volt_mag) * np.exp(1j * np.array(volt_ang) * np.pi / 180)

    from tds_dae_rk_schemes import TDS_simulation

    # Load gt trajectories as initial conditions — this ensures the training
    # data covers the same operating range as inference (including large θ drift)
    gt_files = [
        os.path.join(config_dir, '..', 'gt_simulations', 'sim2s_w_setpoint_3.npy'),
        os.path.join(config_dir, '..', 'gt_simulations', 'sim10s_w_setpoint_3.npy'),
    ]
    gt_states_list = []
    for gf in gt_files:
        if os.path.isfile(gf):
            gt_states_list.append(np.load(gf)[:, :-1])   # [T, 30]
    gt_all = np.vstack(gt_states_list)   # [T_total, 30]

    # state index for machine (0-based):
    # full state: [Eq',Ed',d,w,Id,Iq,Id_g,Iq_g,Vm,Theta] × 3  = 30 vars
    m = machine_idx - 1   # 0-based

    rng  = np.random.default_rng(42)
    inputs_list  = []
    labels_list  = []

    # Sample initial conditions from gt trajectory + small perturbations
    # This covers the full operating range seen during inference
    # CRITICAL: Newton iteration can explore omega values up to ±0.01, which is
    # much larger than normal trajectory range (~±0.004). We add larger omega
    # perturbations to ensure the PINN model covers the full exploration range.
    gt_indices = rng.integers(0, len(gt_all), size=n_traj)
    for traj in tqdm(range(n_traj), desc='Generating trajectories'):
        ic = torch.tensor(gt_all[gt_indices[traj]], dtype=torch.float64)
        # additional perturbation for diversity
        # EXPANDED: omega perturbation from ±0.002 to ±0.008 to cover Newton exploration
        for g in range(3):
            ic[g * 10 + 2] += rng.uniform(-0.02, 0.02)   # delta
            ic[g * 10 + 3] += rng.uniform(-0.008, 0.008)  # omega (expanded!)

        solver = TDS_simulation(
            dampings, freq, H_vec, Xdp_vec, Yadm, pg_pf,
            ic, t_final=n_steps * h, step_size=h,
        )
        t_arr, states = solver.simulation_main_loop(integration_scheme='trapezoidal')
        # states: [n_steps+1, 30]

        for t in range(len(t_arr) - 1):
            x0 = states[t]
            x1 = states[t + 1]

            # ── build PINN input (6-dim) ──────────────────────────────
            Vm0     = x0[m * 10 + 8]
            Vm1     = x1[m * 10 + 8]
            Th0     = x0[m * 10 + 9]
            Th1     = x1[m * 10 + 9]
            d0      = x0[m * 10 + 2]
            d1      = x1[m * 10 + 2]
            om0     = x0[m * 10 + 3]

            theta_pend = (Th1 - Th0) / h
            omicron_0  = d0 - Th0
            # omicron_1 removed - it's not available during inference!

            inp = np.array([Vm0, Vm1, theta_pend, omicron_0, om0, h],
                           dtype=np.float64)

            # ── build label (2-dim): d(omicron)/dt and d(omega)/dt ────────
            # Plug 硬约束: delta1 = h*d_omicron + omicron_0 + theta_pend*h + Theta0
            # PINN 输出1 = d(omicron)/dt = d(delta)/dt - d(Theta)/dt
            #            = (delta1 - delta0)/h - (Theta1 - Theta0)/h
            #            = (delta1 - delta0 - Theta1 + Theta0) / h
            d_omicron = (d1 - d0 - Th1 + Th0) / h
            # PINN 输出2 = d(omega)/dt
            d_omega = (x1[m * 10 + 3] - om0) / h

            label = np.array([d_omicron, d_omega], dtype=np.float64)

            inputs_list.append(inp)
            labels_list.append(label)

    # ── Data Augmentation: enlarge theta_pend and omega coverage ────────────────────────
    # Newton iteration in plug can produce theta_pend and omega values far outside normal
    # trajectory range. To make PINN robust, we augment with synthetic samples covering:
    #   - theta_pend ∈ [-π, π] (same as plug's pretrained model)
    #   - omega ∈ [-0.015, 0.015] (expanded from trajectory range ~[-0.004, 0.004])
    #
    # Strategy: for each normal sample, create augmented copies with perturbed
    # theta_pend and omega. The physics equations remain valid, so we recompute labels.
    n_aug = len(inputs_list) // 2  # augment 50% of original data
    aug_indices = rng.choice(len(inputs_list), size=n_aug, replace=False)
    
    for idx in aug_indices:
        inp_orig = inputs_list[idx].copy()
        Vm0, Vm1, _, omicron_0, omega0_orig, h_val = inp_orig
        
        # Sample theta_pend from [-π, π] to match original plug model's training data distribution
        # Original model's theta_stats: [[-3.138, 3.263], ...]
        theta_pend_aug = rng.uniform(-3.14159, 3.14159)
        # Sample omega from [-0.015, 0.015] to match original model's omega range
        omega0_aug = rng.uniform(-0.015, 0.015)
        
        # For augmented theta_pend and omega, compute d_omicron using physics:
        # d(delta)/dt = omega * 2πf
        freq_val = freq  # 60 Hz
        d_delta_phys = omega0_aug * 2.0 * np.pi * freq_val  # δ rate from swing eq
        # d_omicron = d(delta)/dt - d(Theta)/dt = d_delta_phys - theta_pend
        d_omicron_aug = d_delta_phys - theta_pend_aug
        
        # d(omega)/dt from swing equation (simplified)
        # d(omega)/dt = (Pg - Pe - D*omega) / (2H)
        # For large omega, d(omega)/dt is approximately proportional to omega
        d_omega_aug = labels_list[idx][1]  # keep original d_omega approximation
        
        inp_aug = np.array([Vm0, Vm1, theta_pend_aug, omicron_0, omega0_aug, h_val])
        label_aug = np.array([d_omicron_aug, d_omega_aug])
        
        inputs_list.append(inp_aug)
        labels_list.append(label_aug)

    X = np.stack(inputs_list).astype(np.float64)  # [N, 6]
    Y = np.stack(labels_list).astype(np.float64)  # [N, 2]
    print(f"Generated {len(X)} training pairs ({n_traj * n_steps} normal + {n_aug} augmented)")
    print(f"  theta_pend range after augmentation: [{X[:, 2].min():.4f}, {X[:, 2].max():.4f}]")
    return X, Y


# =============================================================================
# Physics equations  f(x, y)  for single machine
# dδ/dt = ω * 2πf
# dω/dt = (Pg - Pe - D*ω) / (2H)   where Pe = Eq'*Iq + Ed'*Id
# =============================================================================

def physics_f(delta, omega, Eq_prime, Ed_prime, Vm, Theta, machine_params, device):
    """
    Compute [dδ/dt, dω/dt] from physical equations.
    All inputs: [batch, 1] tensors.
    machine_params: dict with H, D, Pg, Xd_p, freq
    """
    freq = machine_params['freq']
    H    = machine_params['H']
    D    = machine_params['D']
    Pg   = machine_params['Pg']
    Xd_p = machine_params['Xd_p']

    # stator equations
    Id = (Eq_prime - Vm * torch.cos(delta - Theta)) / Xd_p
    Iq = -(Ed_prime - Vm * torch.sin(delta - Theta)) / Xd_p
    Pe = Eq_prime * Iq + Ed_prime * Id

    f_delta = omega * 2.0 * np.pi * freq
    f_omega = (Pg - Pe - D * omega) / (2.0 * H)
    return f_delta, f_omega


# =============================================================================
# Compute normalization ranges from data
# =============================================================================

def compute_norm_ranges(X):
    """
    range_norm = [range_list, lb_list]
    where range = max - min, lb = min  (per feature)
    """
    lb    = X.min(axis=0).tolist()
    hi    = X.max(axis=0).tolist()
    rnge  = [hi[i] - lb[i] if hi[i] != lb[i] else 1.0 for i in range(X.shape[1])]
    return rnge, lb


# =============================================================================
# Training loop
# =============================================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"\nDevice: {device}")

    config_dir = os.path.join(_ROOT, 'plug', 'config_files')

    # ---- generate / load data ----
    data_cache = os.path.join(args.log_dir, f'train_data_m{args.machine}.npz')
    os.makedirs(args.log_dir, exist_ok=True)
    if os.path.exists(data_cache) and not args.regen_data:
        print(f"Loading cached data from {data_cache}")
        npz = np.load(data_cache)
        X_all, Y_all = npz['X'], npz['Y']
    else:
        print("Generating training data via TDS simulation...")
        X_all, Y_all = generate_training_data(
            machine_idx=args.machine,
            h=args.h,
            n_traj=args.n_traj,
            n_steps=args.n_steps,
            config_dir=config_dir,
        )
        np.savez(data_cache, X=X_all, Y=Y_all)
        print(f"Data saved to {data_cache}")

    # ---- split train / test ----
    n = len(X_all)
    n_test = max(1, int(n * 0.1))
    idx = np.random.default_rng(0).permutation(n)
    X_tr, Y_tr = X_all[idx[n_test:]], Y_all[idx[n_test:]]
    X_te, Y_te = X_all[idx[:n_test]],  Y_all[idx[:n_test]]

    X_tr_t = torch.tensor(X_tr, dtype=torch.float64).to(device)
    Y_tr_t = torch.tensor(Y_tr, dtype=torch.float64).to(device)
    X_te_t = torch.tensor(X_te, dtype=torch.float64).to(device)
    Y_te_t = torch.tensor(Y_te, dtype=torch.float64).to(device)
    print(f"Train: {len(X_tr)}  Test: {len(X_te)}")

    # ---- normalization ranges ----
    range_norm, lb_norm = compute_norm_ranges(X_all)
    print(f"range_norm: {[f'{r:.4f}' for r in range_norm]}")
    print(f"lb_norm:    {[f'{l:.4f}' for l in lb_norm]}")

    # ---- build model ----
    ModelClass = FCN_RESNET if args.architecture == 'resnet' else FCN_simple
    model = ModelClass(
        N_INPUT=6, N_OUTPUT=2,
        N_HIDDEN=args.n_hidden,
        N_LAYERS=args.n_layers,
        range_states=range_norm,
        lb_states=lb_norm,
        device=str(device),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params} params  (architecture={args.architecture}, hidden={args.n_hidden}, layers={args.n_layers})")

    # ---- optimizer ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.patience, factor=args.factor, verbose=True)

    # ---- training ----
    best_loss = float('inf')
    losses = []
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.log_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    start_time = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'='*60}")
    print(f"[{start_time}] Starting training")
    print(f"Run dir: {run_dir}")
    print(f"{'='*60}\n")

    # ---- load machine physics params ----
    import yaml
    with open(os.path.join(config_dir, 'config_machines_dynamic.yaml')) as f:
        dyn = yaml.safe_load(f)
    m = args.machine - 1
    mparams = {
        'freq': float(dyn['freq']),
        'H':    torch.tensor(list(dyn['inertia_H'].values())[m],   dtype=torch.float64).to(device),
        'D':    torch.tensor(list(dyn['Damping_D'].values())[m],   dtype=torch.float64).to(device),
        'Pg':   torch.tensor(list(dyn['Pg_setpoints'].values())[m],dtype=torch.float64).to(device),
        'Xd_p': torch.tensor(list(dyn['Xd_prime'].values())[m],   dtype=torch.float64).to(device),
    }

    # ---- prepare tensors needed for physics loss ----
    # From X columns: [Vm0, Vm1, theta_pend, omicron_0, omega_0, h]
    # We also need Eq', Ed' from the full state — stored alongside X as aux data
    # For now approximate Eq' ≈ 1.0, Ed' ≈ 0.0 (classical model, constant flux)
    # This is consistent with dEq/dt=0, dEd/dt=0 in the DAE-PINNs formulation

    criterion = nn.MSELoss()
    alpha = args.alpha   # physics loss weight

    for epoch in tqdm(range(args.epochs)):
        model.train()

        # ── mini-batch ──────────────────────────────────────────────────────
        if args.batch_size and args.batch_size < len(X_tr):
            idx_b = torch.randperm(len(X_tr))[:args.batch_size]
            Xb = X_tr_t[idx_b]
            Yb = Y_tr_t[idx_b]   # true derivatives [dδ/dt, dω/dt]
        else:
            Xb, Yb = X_tr_t, Y_tr_t

        optimizer.zero_grad()

        # ── Data Loss: hard-constraint prediction vs true next state ─────────
        # Input layout: [Vm0, Vm1, theta_pend, omicron_0, omega_0, h]
        deriv_pred = model(Xb)             # [batch, 2]  predicted derivatives
        loss_data  = criterion(deriv_pred, Yb)

        # ── Physics Loss: AutoDiff residual at random τ ∈ [0, h] ────────────
        loss_phys = torch.tensor(0.0, device=device)
        if alpha > 0:
            # τ as a learnable input dimension: replace h-column with τ
            h_col   = Xb[:, 5:6]                           # [batch, 1]  (h is now column 5)
            tau     = torch.rand_like(h_col) * h_col       # τ ∈ [0, h]
            tau.requires_grad_(True)

            # build input with τ instead of h
            Xb_tau  = torch.cat([Xb[:, :5], tau], dim=1)   # [batch, 6]
            deriv_tau = model(Xb_tau)                       # [batch, 2]  NN output at τ

            # d(deriv_tau)/d(tau) via AutoDiff
            d_delta_dtau = torch.autograd.grad(
                deriv_tau[:, 0].sum(), tau,
                create_graph=True, retain_graph=True
            )[0]                                            # [batch, 1]
            d_omega_dtau = torch.autograd.grad(
                deriv_tau[:, 1].sum(), tau,
                create_graph=True
            )[0]                                            # [batch, 1]

            # physics rhs at τ: reconstruct state at τ using linear interp
            # PINN 输出: d(omicron)/dt = d(delta)/dt - d(Theta)/dt
            #           d(omega)/dt
            # 状态在 τ 时刻:
            #   omicron_τ = omicron_0 + d(omicron)/dt * τ
            #   omega_τ = omega_0 + d(omega)/dt * τ
            Vm0       = Xb[:, 0:1]
            Vm1       = Xb[:, 1:2]
            theta_p   = Xb[:, 2:3]   # dθ/dt = (Theta1 - Theta0)/h
            omicron_0 = Xb[:, 3:4]
            omega_0   = Xb[:, 4:5]

            # linear interp voltage
            h_col_det = h_col.detach()
            Vm_tau    = Vm0 + (Vm1 - Vm0) * (tau / (h_col_det + 1e-8))
            # Theta at τ: Theta_τ = Theta_0 + theta_p * τ  (假设 Theta_0 ≈ 0)
            Th_tau    = theta_p * tau
            # delta_τ = omicron_τ + Theta_τ = omicron_0 + d(omicron)/dt * τ + Theta_τ
            delta_tau = omicron_0 + deriv_tau[:, 0:1] * tau + Th_tau
            omega_tau = omega_0   + deriv_tau[:, 1:2] * tau

            # classical model: Eq'≈1, Ed'≈0
            Eq_p = torch.ones_like(delta_tau)
            Ed_p = torch.zeros_like(delta_tau)

            # physics_f 返回: d(delta)/dt 和 d(omega)/dt
            f_delta_dt, f_omega_dt = physics_f(delta_tau, omega_tau, Eq_p, Ed_p,
                                  Vm_tau, Th_tau, mparams, device)

            # PINN 输出的导数应该满足:
            # d(omicron)/dt = d(delta)/dt - d(Theta)/dt
            # d(omega)/dt = physics 方程
            f_omicron_dt = f_delta_dt - theta_p  # d(delta)/dt - d(Theta)/dt

            # residual: d(NN)/dτ 应该与物理方程一致
            loss_phys = criterion(d_delta_dtau, f_omicron_dt) + criterion(d_omega_dtau, f_omega_dt)

        total_loss = loss_data + alpha * loss_phys
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())
        scheduler.step(total_loss)

        if (epoch + 1) % args.test_every == 0:
            model.eval()
            with torch.no_grad():
                pred_te  = model(X_te_t)
                loss_te  = criterion(pred_te, Y_te_t).item()
            now_str = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n[{now_str}] Epoch {epoch+1:>6d} | data={loss_data.item():.3e}"
                  f"  phys={loss_phys.item() if alpha>0 else 0:.3e}"
                  f"  test={loss_te:.3e}")
            model.train()

            if loss_te < best_loss:
                best_loss = loss_te
                _save_checkpoint(model, run_dir, args, range_norm, lb_norm,
                                 epoch, losses, is_best=True)

        if (epoch + 1) % args.save_every == 0:
            _save_checkpoint(model, run_dir, args, range_norm, lb_norm,
                             epoch, losses, is_best=False)

    end_time = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'='*60}")
    print(f"[{end_time}] Training completed!")
    print(f"Best test loss: {best_loss:.4e}")
    print(f"Model saved to: {run_dir}")
    print(f"{'='*60}")
    np.save(os.path.join(run_dir, 'losses.npy'), np.array(losses))


# =============================================================================
# Checkpoint saving (plug-compatible format)
# =============================================================================

def _save_checkpoint(model, run_dir, args, range_norm, lb_norm, epoch, losses, is_best):
    """Save in the exact format plug/main.py expects."""
    # load physics params for metadata
    config_dir = os.path.join(_ROOT, 'plug', 'config_files')
    with open(os.path.join(config_dir, 'config_machines_dynamic.yaml'), 'r') as f:
        dyn = yaml.safe_load(f)
    with open(os.path.join(config_dir, 'config_machines_static.yaml'), 'r') as f:
        sta = yaml.safe_load(f)

    m = args.machine - 1
    H_list   = list(dyn['inertia_H'].values())
    Xdp_list = list(dyn['Xd_prime'].values())
    pg_list  = list(dyn['Pg_setpoints'].values())
    D_list   = list(dyn['Damping_D'].values())
    freq     = dyn['freq']

    # machine_parameters: (D, Pg, H, Xd_p, freq)  — order from plug load_pinn_parameters
    machine_params = (D_list[m], pg_list[m], H_list[m], Xdp_list[m], freq)

    # architecture: [N_HIDDEN, N_LAYERS, N_INPUT, N_OUTPUT] for FCN_simple
    #              or [N_HIDDEN, N_HIDDEN_RES, N_LAYERS, N_INPUT, N_OUTPUT] for ResNet
    if args.architecture == 'resnet':
        arch = [args.n_hidden, args.n_hidden, args.n_layers, 6, 2]
    else:
        arch = [args.n_hidden, args.n_layers, 6, 2]

    # operating limits (from range_norm data)
    voltage_lo  = float(lb_norm[0])
    voltage_hi  = float(lb_norm[0] + range_norm[0])
    voltage_lo2 = float(lb_norm[1])
    voltage_hi2 = float(lb_norm[1] + range_norm[1])
    theta_lo    = float(lb_norm[2])
    theta_hi    = float(lb_norm[2] + range_norm[2])
    delta_lo    = float(lb_norm[3])
    delta_hi    = float(lb_norm[3] + range_norm[3])
    omega_lo    = float(lb_norm[4])
    omega_hi    = float(lb_norm[4] + range_norm[4])

    ckpt = {
        'state_dict':         model.state_dict(),
        'epochs':             epoch + 1,
        'col_points':         [args.h, args.n_steps, len(losses)],
        'architecture':       arch,
        'architecture_type':  args.architecture,  # 'resnet' or 'fcn'
        'init_state':         [None, None, [delta_lo, delta_hi], [omega_lo, omega_hi]],
        'range_norm':         [range_norm, lb_norm],
        'range_unorm':        [[0.0, 0.0], [0.0, 0.0]],   # placeholder
        'voltage_stats':      [[voltage_lo, voltage_hi], [voltage_lo2, voltage_hi2],
                               args.n_hidden],
        'theta_stats':        [[theta_lo, theta_hi], [theta_lo, theta_hi], args.n_hidden],
        'machine_parameters': machine_params,
        'losses':             losses,
        'training_info':      [args.n_hidden, args.lr, 1.0, args.epochs,
                               'adam', args.factor, args.epochs],
    }

    fname = 'model_best.pth' if is_best else f'model_epoch{epoch+1}.pth'
    path  = os.path.join(run_dir, fname)
    torch.save(ckpt, path)
    if is_best:
        print(f"  → Best model saved: {path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser('Train plug-compatible single-machine PINN')
    p.add_argument('--machine',    type=int,   default=1,       help='Generator index (1/2/3)')
    p.add_argument('--h',          type=float, default=0.05,    help='Time step for simulation')
    p.add_argument('--n_traj',     type=int,   default=500,     help='Number of trajectories')
    p.add_argument('--n_steps',    type=int,   default=50,      help='Steps per trajectory')
    p.add_argument('--regen_data', action='store_true',          help='Re-generate data even if cache exists')
    p.add_argument('--architecture', type=str, default='resnet', choices=['resnet', 'fcn'],
                    help='Network architecture: resnet (default, matches plug) or fcn')
    p.add_argument('--n_hidden',   type=int,   default=128,     help='Hidden layer size')
    p.add_argument('--n_layers',   type=int,   default=3,       help='Number of hidden layers')
    p.add_argument('--epochs',     type=int,   default=50000,   help='Training epochs')
    p.add_argument('--lr',         type=float, default=5e-4,    help='Learning rate')
    p.add_argument('--batch_size', type=int,   default=4096,    help='Batch size')
    p.add_argument('--patience',   type=int,   default=2000,    help='LR scheduler patience')
    p.add_argument('--factor',     type=float, default=0.5,     help='LR scheduler factor')
    p.add_argument('--test_every', type=int,   default=500,     help='Test interval')
    p.add_argument('--save_every', type=int,   default=5000,    help='Save interval')
    p.add_argument('--log_dir',    type=str,   default='./logs/plug_pinn', help='Log directory')
    p.add_argument('--alpha',      type=float, default=1.0,     help='Physics loss weight')
    p.add_argument('--device',     type=str,   default='cuda',  help='cuda or cpu')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
