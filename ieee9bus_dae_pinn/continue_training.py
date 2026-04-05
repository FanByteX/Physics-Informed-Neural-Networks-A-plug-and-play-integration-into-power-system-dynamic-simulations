"""
Continue training from a checkpoint with more epochs.
Automatically detects the latest model and continues training.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Set float64 globally
torch.set_default_dtype(torch.float64)

# Use local modules (independent of plug)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'src'))

# Import model classes and physics function from train_plug_pinn.py
from train_plug_pinn import FCN_RESNET, FCN_simple, physics_f
config_dir = os.path.join(_HERE, 'config_files')


def find_latest_model(log_dir):
    """Find the most recent model directory by file modification time."""
    if not os.path.exists(log_dir):
        return None, None
    
    subdirs = [d for d in os.listdir(log_dir) 
               if os.path.isdir(os.path.join(log_dir, d))]
    if not subdirs:
        return None, None
    
    subdirs_with_mtime = [(d, os.path.getmtime(os.path.join(log_dir, d))) for d in subdirs]
    subdirs_with_mtime.sort(key=lambda x: x[1], reverse=True)
    
    import datetime
    latest = subdirs_with_mtime[0][0]
    mtime = datetime.datetime.fromtimestamp(subdirs_with_mtime[0][1])
    print(f"Latest run: {latest} (modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
    
    return os.path.join(log_dir, latest), os.path.join(log_dir, latest, 'model_best.pth')


def load_checkpoint(model_path, device):
    """Load model from checkpoint."""
    ckpt = torch.load(model_path, map_location=device)
    
    arch = ckpt['architecture']
    arch_type = ckpt.get('architecture_type', 'resnet')
    
    if arch_type == 'resnet':
        n_hidden = arch[0]
        n_layers = arch[2]
    else:
        n_hidden = arch[0]
        n_layers = arch[1]
    
    print(f"Model architecture: {arch_type}, hidden={n_hidden}, layers={n_layers}")
    print(f"Checkpoint epochs: {ckpt.get('epochs', 'unknown')}")
    
    return ckpt, n_hidden, n_layers, arch_type


def load_training_data(data_cache_path, ckpt):
    """Load cached training data. Uses range_norm and lb_norm from checkpoint if not in data file."""
    data = np.load(data_cache_path)
    X = data['X']
    Y = data['Y']
    
    # Get range_norm and lb_norm from checkpoint if not in data file
    if 'range_norm' in data:
        range_norm = data['range_norm']
        lb_norm = data['lb_norm']
    else:
        range_norm = ckpt.get('range_norm', [[0, 0], [1, 1]])
        lb_norm = ckpt.get('lb_norm', None)
        if lb_norm is None:
            # Estimate from data if needed
            lb_norm = [X.min(axis=0).tolist(), X.max(axis=0).tolist()]
    
    return X, Y, range_norm, lb_norm


def main():
    parser = argparse.ArgumentParser('Continue training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (auto-detect if not specified)')
    parser.add_argument('--additional_epochs', type=int, default=100000, help='Additional epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')
    parser.add_argument('--alpha', type=float, default=1.0, help='Physics loss weight')
    parser.add_argument('--patience', type=int, default=5000, help='LR scheduler patience')
    parser.add_argument('--factor', type=float, default=0.5, help='LR scheduler factor')
    parser.add_argument('--test_every', type=int, default=1000, help='Test interval')
    parser.add_argument('--save_every', type=int, default=10000, help='Save interval')
    parser.add_argument('--log_dir', type=str, default='./logs/plug_pinn', help='Log directory')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Find checkpoint
    if args.checkpoint:
        run_dir = os.path.dirname(args.checkpoint)
        model_path = args.checkpoint
    else:
        run_dir, model_path = find_latest_model(args.log_dir)
        if run_dir is None:
            print("No checkpoint found!")
            return
    
    print(f"Checkpoint: {model_path}")
    
    # Load checkpoint
    ckpt, n_hidden, n_layers, arch_type = load_checkpoint(model_path, device)
    
    # Load training data
    data_cache = os.path.join(args.log_dir, 'train_data_m1.npz')
    if not os.path.exists(data_cache):
        print(f"Training data not found: {data_cache}")
        return
    
    X, Y, range_norm, lb_norm = load_training_data(data_cache, ckpt)
    print(f"Training data: {X.shape[0]} samples")
    
    # Train/test split
    n_train = int(0.9 * len(X))
    X_tr, X_te = X[:n_train], X[n_train:]
    Y_tr, Y_te = Y[:n_train], Y[n_train:]
    
    X_tr_t = torch.from_numpy(X_tr).to(device)
    Y_tr_t = torch.from_numpy(Y_tr).to(device)
    X_te_t = torch.from_numpy(X_te).to(device)
    Y_te_t = torch.from_numpy(Y_te).to(device)
    
    # Create model - use range_states and lb_states from checkpoint state_dict
    # These are stored as 1D tensors in state_dict, matching nn.Parameter format
    range_states_ckpt = ckpt['state_dict'].get('range_states', None)
    lb_states_ckpt = ckpt['state_dict'].get('lb_states', None)
    
    if arch_type == 'resnet':
        model = FCN_RESNET(
            N_INPUT=6, N_OUTPUT=2,
            N_HIDDEN=n_hidden,
            N_LAYERS=n_layers,
            range_states=range_states_ckpt,
            lb_states=lb_states_ckpt,
            device=str(device),
        ).to(device)
    else:
        model = FCN_simple(
            N_INPUT=6, N_OUTPUT=2,
            N_HIDDEN=n_hidden,
            N_LAYERS=n_layers,
            range_states=range_states_ckpt,
            lb_states=lb_states_ckpt,
            device=str(device),
        ).to(device)
    
    model.load_state_dict(ckpt['state_dict'])
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load previous losses
    prev_losses = ckpt.get('losses', [])
    if not prev_losses:
        losses_path = os.path.join(run_dir, 'losses.npy')
        if os.path.exists(losses_path):
            prev_losses = np.load(losses_path).tolist()
    
    start_epoch = len(prev_losses)
    print(f"Starting from epoch {start_epoch}")
    
    # Optimizer with lower LR for fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.patience, factor=args.factor, verbose=True)
    
    # Load physics params
    import yaml
    with open(os.path.join(config_dir, 'config_machines_dynamic.yaml')) as f:
        dyn = yaml.safe_load(f)
    
    mparams = {
        'freq': float(dyn['freq']),
        'H':    torch.tensor(list(dyn['inertia_H'].values())[0], dtype=torch.float64).to(device),
        'D':    torch.tensor(list(dyn['Damping_D'].values())[0], dtype=torch.float64).to(device),
        'Pg':   torch.tensor(list(dyn['Pg_setpoints'].values())[0], dtype=torch.float64).to(device),
        'Xd_p': torch.tensor(list(dyn['Xd_prime'].values())[0], dtype=torch.float64).to(device),
    }
    
    criterion = nn.MSELoss()
    losses = prev_losses.copy()
    best_loss = float(min(prev_losses)) if prev_losses else float('inf')
    
    # Continue in same directory
    start_time = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'='*60}")
    print(f"[{start_time}] Continuing training from epoch {start_epoch}")
    print(f"Target epochs: {start_epoch + args.additional_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*60}\n")
    
    for epoch in tqdm(range(start_epoch, start_epoch + args.additional_epochs)):
        model.train()
        
        # Mini-batch
        if args.batch_size < len(X_tr):
            idx_b = torch.randperm(len(X_tr))[:args.batch_size]
            Xb = X_tr_t[idx_b]
            Yb = Y_tr_t[idx_b]
        else:
            Xb, Yb = X_tr_t, Y_tr_t
        
        optimizer.zero_grad()
        
        # Data loss
        deriv_pred = model(Xb)
        loss_data = criterion(deriv_pred, Yb)
        
        # Physics loss
        loss_phys = torch.tensor(0.0, device=device)
        if args.alpha > 0:
            h_col = Xb[:, 5:6]
            tau = torch.rand_like(h_col) * h_col
            tau.requires_grad_(True)
            
            Xb_tau = torch.cat([Xb[:, :5], tau], dim=1)
            deriv_tau = model(Xb_tau)
            
            d_delta_dtau = torch.autograd.grad(
                deriv_tau[:, 0].sum(), tau,
                create_graph=True, retain_graph=True
            )[0]
            d_omega_dtau = torch.autograd.grad(
                deriv_tau[:, 1].sum(), tau,
                create_graph=True
            )[0]
            
            Vm0 = Xb[:, 0:1]
            Vm1 = Xb[:, 1:2]
            theta_p = Xb[:, 2:3]
            omicron_0 = Xb[:, 3:4]
            omega_0 = Xb[:, 4:5]
            
            h_col_det = h_col.detach()
            Vm_tau = Vm0 + (Vm1 - Vm0) * (tau / (h_col_det + 1e-8))
            Th_tau = theta_p * tau
            delta_tau = omicron_0 + deriv_tau[:, 0:1] * tau + Th_tau
            omega_tau = omega_0 + deriv_tau[:, 1:2] * tau
            
            Eq_p = torch.ones_like(delta_tau)
            Ed_p = torch.zeros_like(delta_tau)
            
            f_delta_dt, f_omega_dt = physics_f(delta_tau, omega_tau, Eq_p, Ed_p,
                                              Vm_tau, Th_tau, mparams, device)
            
            f_omicron_dt = f_delta_dt - theta_p
            loss_phys = criterion(d_delta_dtau, f_omicron_dt) + criterion(d_omega_dtau, f_omega_dt)
        
        total_loss = loss_data + args.alpha * loss_phys
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())
        scheduler.step(total_loss)
        
        if (epoch + 1) % args.test_every == 0:
            model.eval()
            with torch.no_grad():
                pred_te = model(X_te_t)
                loss_te = criterion(pred_te, Y_te_t).item()
            
            now_str = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n[{now_str}] Epoch {epoch+1:>6d} | data={loss_data.item():.3e}"
                  f"  phys={loss_phys.item() if args.alpha>0 else 0:.3e}"
                  f"  test={loss_te:.3e}")
            model.train()
            
            if loss_te < best_loss:
                best_loss = loss_te
                # Save best model
                ckpt['state_dict'] = model.state_dict()
                ckpt['epochs'] = epoch + 1
                ckpt['losses'] = losses
                torch.save(ckpt, os.path.join(run_dir, 'model_best.pth'))
                print(f"  → Best model updated: test={loss_te:.6e}")
        
        if (epoch + 1) % args.save_every == 0:
            # Save checkpoint
            ckpt['state_dict'] = model.state_dict()
            ckpt['epochs'] = epoch + 1
            ckpt['losses'] = losses
            torch.save(ckpt, os.path.join(run_dir, f'model_epoch{epoch+1}.pth'))
            
            # Also save losses
            np.save(os.path.join(run_dir, 'losses.npy'), np.array(losses))
    
    # Final save
    end_time = time.strftime('%Y-%m-%d %H:%M:%S')
    ckpt['state_dict'] = model.state_dict()
    ckpt['epochs'] = len(losses)
    ckpt['losses'] = losses
    torch.save(ckpt, os.path.join(run_dir, 'model_final.pth'))
    np.save(os.path.join(run_dir, 'losses.npy'), np.array(losses))
    
    print(f"\n{'='*60}")
    print(f"[{end_time}] Training completed!")
    print(f"Total epochs: {len(losses)}")
    print(f"Best test loss: {best_loss:.4e}")
    print(f"Model saved to: {run_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
