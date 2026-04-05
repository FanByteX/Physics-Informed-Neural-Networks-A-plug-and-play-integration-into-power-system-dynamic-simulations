"""
Automatic hyperparameter optimization for plug PINN training.

Analyzes current model performance and automatically:
1. Diagnoses issues (underfitting, overfitting, etc.)
2. Suggests and applies parameter adjustments
3. Retrains and validates improvement
"""

import os
import sys
import subprocess
import numpy as np
import torch
from datetime import datetime

# Set float64 globally
torch.set_default_dtype(torch.float64)

# Use local modules (independent of plug)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'src'))


def analyze_training(run_dir):
    """Analyze training results to diagnose issues."""
    losses_path = os.path.join(run_dir, 'losses.npy')
    if not os.path.exists(losses_path):
        return None
    
    losses = np.load(losses_path)
    
    # Analyze loss curve
    n_epochs = len(losses)
    final_loss = losses[-1]
    best_loss = losses.min()
    best_epoch = np.argmin(losses)
    
    # Check if still improving
    recent_1000 = losses[-1000:] if n_epochs >= 1000 else losses[-100:]
    slope = (recent_1000[-1] - recent_1000[0]) / len(recent_1000)
    
    # Check for plateau
    plateau_start = None
    for i in range(n_epochs - 1, max(0, n_epochs - 5000), -1):
        if abs(losses[i] - best_loss) < 0.01 * best_loss:
            continue
        else:
            plateau_start = i
            break
    
    return {
        'n_epochs': n_epochs,
        'final_loss': final_loss,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'slope': slope,
        'still_improving': slope < -1e-8,
        'plateau_epoch': plateau_start,
    }


def run_inference(model_path, study_selection=2):
    """Run inference and return error metrics."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_plug_inference", 
        os.path.join(os.path.dirname(__file__), "run_plug_inference.py")
    )
    inference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference_module)
    
    # Run simulation
    cmd = [
        'python', 'run_plug_inference.py',
        '--pinn_path', model_path,
        '--machine', '1',
        '--study_selection', str(study_selection),
        '--compare_ground_truth',
        '--out_dir', './outputs_optim'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
    
    # Load and compute errors
    sim_path = os.path.join(os.path.dirname(__file__), 'outputs_optim', 'sim_results.npz')
    if os.path.exists(sim_path):
        from scipy.interpolate import interp1d
        data = np.load(sim_path)
        t_pinn, s_pinn = data['t_pinn'], data['s_pinn']
        t_true, s_true = data['t_true'], data['s_true']
        
        # Interpolate true solution
        s_true_interp = np.zeros_like(s_pinn)
        for i in range(s_true.shape[1]):
            f = interp1d(t_true, s_true[:, i], kind='cubic')
            s_true_interp[:, i] = f(t_pinn)
        
        # Compute RMSE
        rmse = np.sqrt(np.mean((s_pinn - s_true_interp)**2))
        return {'rmse': rmse, 'success': True}
    
    return {'rmse': float('inf'), 'success': False}


def suggest_improvements(analysis, current_params):
    """Suggest parameter improvements based on analysis."""
    suggestions = {}
    
    if analysis is None:
        return {'epochs': 100000, 'lr': 1e-4, 'n_hidden': 256, 'n_layers': 4}
    
    # If still improving, train longer
    if analysis['still_improving']:
        suggestions['epochs'] = current_params.get('epochs', 50000) + 50000
        print(f"  → Training still improving, suggest more epochs: {suggestions['epochs']}")
    
    # If loss is high, increase model capacity
    if analysis['best_loss'] > 1e-4:
        suggestions['n_hidden'] = min(current_params.get('n_hidden', 128) * 2, 512)
        suggestions['n_layers'] = min(current_params.get('n_layers', 3) + 1, 6)
        print(f"  → High loss, suggest larger model: hidden={suggestions.get('n_hidden')}, layers={suggestions.get('n_layers')}")
    
    # If plateaued but not converged, adjust learning rate
    if analysis['slope'] > -1e-8 and analysis['best_loss'] > 1e-5:
        current_lr = current_params.get('lr', 5e-4)
        suggestions['lr'] = current_lr * 0.5  # Lower LR for fine-tuning
        print(f"  → Plateau reached, suggest lower LR: {suggestions['lr']}")
    
    return suggestions


def train_with_params(params, run_name):
    """Run training with specified parameters."""
    cmd = ['python', 'train_plug_pinn.py']
    cmd.extend(['--epochs', str(params.get('epochs', 100000))])
    cmd.extend(['--lr', str(params.get('lr', 5e-4))])
    cmd.extend(['--n_hidden', str(params.get('n_hidden', 128))])
    cmd.extend(['--n_layers', str(params.get('n_layers', 3))])
    cmd.extend(['--batch_size', str(params.get('batch_size', 4096))])
    cmd.extend(['--alpha', str(params.get('alpha', 1.0))])
    cmd.extend(['--patience', str(params.get('patience', 3000))])
    cmd.extend(['--n_traj', str(params.get('n_traj', 500))])
    cmd.extend(['--n_steps', str(params.get('n_steps', 50))])
    
    print(f"\n{'='*60}")
    print(f"Training with params: {params}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    return result.returncode == 0


def find_latest_model(log_dir):
    """Find the most recent model directory by file modification time."""
    if not os.path.exists(log_dir):
        return None, None
    
    subdirs = [d for d in os.listdir(log_dir) 
               if os.path.isdir(os.path.join(log_dir, d))]
    if not subdirs:
        return None, None
    
    # Sort by modification time (most recent first)
    subdirs_with_mtime = [(d, os.path.getmtime(os.path.join(log_dir, d))) for d in subdirs]
    subdirs_with_mtime.sort(key=lambda x: x[1], reverse=True)
    
    latest = subdirs_with_mtime[0][0]
    model_path = os.path.join(log_dir, latest, 'model_best.pth')
    
    # Also print info
    import datetime
    mtime = datetime.datetime.fromtimestamp(subdirs_with_mtime[0][1])
    print(f"  Latest run: {latest} (modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
    
    return os.path.join(log_dir, latest), model_path


def main():
    print("="*70)
    print("AUTO OPTIMIZATION FOR PLUG PINN")
    print("="*70)
    
    log_dir = './logs/plug_pinn'
    
    # Step 1: Analyze current best model
    print("\n[Step 1] Analyzing current training...")
    run_dir, model_path = find_latest_model(log_dir)
    
    if run_dir is None:
        print("No previous training found. Starting fresh training...")
        params = {
            'epochs': 100000,
            'lr': 5e-4,
            'n_hidden': 256,  # Larger model
            'n_layers': 4,
            'batch_size': 4096,
            'alpha': 1.0,
            'patience': 3000,
        }
    else:
        print(f"Found latest run: {run_dir}")
        analysis = analyze_training(run_dir)
        
        if analysis:
            print(f"\n  Epochs trained: {analysis['n_epochs']}")
            print(f"  Best loss: {analysis['best_loss']:.6e} at epoch {analysis['best_epoch']}")
            print(f"  Final loss: {analysis['final_loss']:.6e}")
            print(f"  Still improving: {analysis['still_improving']}")
            print(f"  Loss slope: {analysis['slope']:.2e}")
        
        # Get current params from checkpoint
        ckpt = torch.load(model_path, map_location='cpu')
        current_params = {
            'epochs': ckpt.get('epochs', 50000),
            'n_hidden': ckpt['architecture'][0],
            'n_layers': ckpt['architecture'][2] if len(ckpt['architecture']) == 5 else ckpt['architecture'][1],
        }
        
        # Run inference to get current error
        print("\n[Step 2] Running inference on current model...")
        current_metrics = run_inference(model_path)
        print(f"  Current RMSE: {current_metrics['rmse']:.6f}")
        
        # Suggest improvements
        print("\n[Step 3] Analyzing and suggesting improvements...")
        suggestions = suggest_improvements(analysis, current_params)
        
        # Merge with current params
        params = current_params.copy()
        params.update(suggestions)
        params['epochs'] = max(params.get('epochs', 100000), 100000)  # At least 100k epochs
    
    # Step 4: Train with improved params
    print("\n[Step 4] Training with optimized parameters...")
    success = train_with_params(params, "optimized")
    
    if success:
        # Step 5: Validate improvement
        print("\n[Step 5] Validating improvement...")
        new_run_dir, new_model_path = find_latest_model(log_dir)
        if new_run_dir != run_dir:  # New training happened
            new_metrics = run_inference(new_model_path)
            print(f"\n  New RMSE: {new_metrics['rmse']:.6f}")
            
            if 'current_metrics' in dir() and new_metrics['rmse'] < current_metrics['rmse']:
                print(f"\n  ✓ IMPROVEMENT: {((current_metrics['rmse'] - new_metrics['rmse']) / current_metrics['rmse'] * 100):.1f}%")
            elif 'current_metrics' in dir():
                print(f"\n  ✗ No improvement, may need different strategy")
    
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
