"""
run_plug_inference.py
=====================
Plug-n-Play inference using a trained single-machine PINN.

Replicates plug/main.py completely, but:
  - Loads the PINN with FCN_simple (matches actual saved weight keys)
  - Uses paths relative to this project
  - Supports the same --study_selection 1-5 as the original paper

Usage:
    cd ieee9bus_dae_pinn
    python run_plug_inference.py --pinn_path ./logs/plug_pinn/<ts>/model_best.pth \
        --machine 1 --study_selection 2 --time_step_size 8e-3 --sim_time 10

    # or use the shipped pretrained model:
    python run_plug_inference.py --machine 1 --study_selection 1
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch

# ── project paths ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

# Use local src modules (independent of plug)
_LOCAL_SRC = os.path.join(_HERE, 'src')
sys.path.insert(0, _LOCAL_SRC)

from tds_dae_rk_schemes import TDS_simulation

# Use local post_processing modules (not plug's)
from post_processing.trajectories_overview_plot import trajectories_overview
from post_processing.custom_overview_plots import custom_overview1, custom_overview2

# our networks that match the saved weight keys
from train_plug_pinn import FCN_simple, FCN_RESNET


class PINNWrapper(torch.nn.Module):
    """
    Wraps FCN to be compatible with plug's pinn_integration_scheme.

    plug passes 6-dim input: [Vm0, Vm1, theta_pend, omicron_0, omega0, h]
    Our model (if trained with N_INPUT=6) expects the same 6-dim input.
    
    For backwards compatibility with plug's pretrained models (7-dim input):
    Pretrained model expects 7-dim:  [..., omicron_1, ...]
    Layout expected by model: [Vm0, Vm1, theta_pend, omicron_0, omicron_1, omega0, h]
    Layout from plug:         [Vm0, Vm1, theta_pend, omicron_0,            omega0, h]

    We insert omicron_1 = omicron_0 (constant approximation during Newton iteration)
    between dim 3 and dim 4.
    """
    def __init__(self, base_model, n_input):
        super().__init__()
        self.base  = base_model
        self.n_in  = n_input   # expected input dim of base model

    def forward(self, x):
        # plug passes float64; cast to model's weight dtype
        dtype = next(self.base.parameters()).dtype
        x = x.to(dtype)

        if self.n_in == 7 and x.shape[-1] == 6:
            # plug sends 6-dim: [Vm0, Vm1, theta_pend, omicron_0, omega0, h]
            # model expects 7-dim: insert omicron_1 = omicron_0 at position 4
            x = torch.cat([x[..., :4], x[..., 3:4], x[..., 4:]], dim=-1)

        # No theta_pend wrapping needed - training data now covers [-π, π]
        # which matches plug's pretrained model range
        return self.base(x)


# =============================================================================
# Config helpers  (identical to plug/main.py)
# =============================================================================

def config_file(yaml_file) -> dict:
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


def extract_values_dynamic_components(cfg) -> tuple:
    freq     = cfg['freq']
    H        = list(cfg['inertia_H'].values())
    Rs       = list(cfg['Rs'].values())
    Xd_p     = list(cfg['Xd_prime'].values())
    pg_pf    = list(cfg['Pg_setpoints'].values())
    dampings = list(cfg['Damping_D'].values())
    return freq, H, Rs, Xd_p, pg_pf, dampings


def extract_values_static_components(cfg) -> tuple:
    volt_mag  = list(cfg['Voltage_magnitude'].values())
    volt_ang  = list(cfg['Voltage_angle'].values())
    Xd        = list(cfg['Xd'].values())
    Xq        = list(cfg['Xq'].values())
    Xq_prime  = list(cfg['Xq_prime'].values())
    volt_cpx  = np.array(volt_mag) * np.exp(1j * np.array(volt_ang) * np.pi / 180)
    return volt_cpx, Xd, Xq, Xq_prime


# =============================================================================
# PINN loading  (auto-detects ResNet vs FCN_simple architecture)
# =============================================================================

def load_pinn_machine(pinn_path, device):
    ckpt = torch.load(pinn_path, map_location=device)
    norm_range, lb_range = ckpt['range_norm']
    arch = ckpt['architecture']
    
    # Detect architecture type from checkpoint
    arch_type = ckpt.get('architecture_type', None)
    
    # Parse architecture based on length
    if len(arch) == 5:
        # ResNet: [N_HIDDEN, N_HIDDEN_RES, N_LAYERS, N_INPUT, N_OUTPUT]
        N_HIDDEN, N_HIDDEN_RES, N_LAYERS, N_INPUT, N_OUTPUT = arch
        if arch_type is None:
            arch_type = 'resnet'  # Infer from arch length
        print(f"Loading ResNet model: hidden={N_HIDDEN}, layers={N_LAYERS}, input={N_INPUT}")
        model = FCN_RESNET(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, norm_range, lb_range)
    elif len(arch) == 4:
        # FCN_simple: [N_HIDDEN, N_LAYERS, N_INPUT, N_OUTPUT]
        N_HIDDEN, N_LAYERS, N_INPUT, N_OUTPUT = arch
        if arch_type is None:
            arch_type = 'fcn'  # Infer from arch length
        print(f"Loading FCN_simple model: hidden={N_HIDDEN}, layers={N_LAYERS}, input={N_INPUT}")
        model = FCN_simple(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, norm_range, lb_range)
    else:
        raise ValueError(f"Unknown architecture format: {arch}")
    
    # buffers (range_states, lb_states) are not in state_dict → strict=False
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    # wrap to handle plug's 6-dim vs model's 7-dim input difference
    wrapped = PINNWrapper(model, n_input=N_INPUT)
    wrapped.to(device)
    wrapped.eval()
    return wrapped


def load_pinn_parameters(pinn_path) -> tuple:
    """Returns (H, Xd_p, Pg, D) for the trained machine."""
    ckpt = torch.load(pinn_path, map_location='cpu')
    D, Pg, H, Xd_p, _ = ckpt['machine_parameters']
    return H, Xd_p, Pg, D


def compute_pinn_ops_limits(pinn_path) -> tuple:
    ckpt = torch.load(pinn_path, map_location='cpu')
    voltage_limits = ckpt['voltage_stats'][0]
    theta_limits   = ckpt['theta_stats'][0]
    delta_limits   = ckpt['init_state'][2]
    omega_limits   = ckpt['init_state'][3]
    return voltage_limits, theta_limits, delta_limits, omega_limits


# =============================================================================
# Ground-truth helpers  (identical to plug/main.py)
# =============================================================================

def load_ini_conditions_true(args, start_value=0):
    gt_dir    = os.path.join(_HERE, 'gt_simulations')
    file_name = f'sim{int(args.sim_time)}s_{args.event_type}_{args.event_location}.npy'
    states    = np.load(os.path.join(gt_dir, file_name))
    return torch.tensor(states[start_value, :-1], dtype=torch.float64)


def load_ini_conditions_true_option4(args, start_value=0):
    gt_dir = os.path.join(_HERE, 'gt_simulations')
    states = np.load(os.path.join(gt_dir,
                     f'sim10s_{args.event_type}_{args.event_location}.npy'))
    return torch.tensor(states[start_value, :-1], dtype=torch.float64)


def return_true_solution(args):
    gt_dir    = os.path.join(_HERE, 'gt_simulations')
    file_name = f'sim{int(args.sim_time)}s_{args.event_type}_{args.event_location}.npy'
    states    = np.load(os.path.join(gt_dir, file_name))
    return states[:, 30], states[:, :-1]


def return_true_solution_option4(args, start=0, end=-1):
    gt_dir = os.path.join(_HERE, 'gt_simulations')
    states = np.load(os.path.join(gt_dir,
                     f'sim10s_{args.event_type}_{args.event_location}.npy'))
    return states[start:end, 30], states[start:end, :-1]


# =============================================================================
# Error analysis  (identical to plug/main.py)
# =============================================================================

def compute_time_step_assimulo(time_array):
    dt = round(time_array[1] - time_array[0], 8)
    for i in np.random.randint(0, len(time_array), size=10).tolist():
        assert round(time_array[i] - time_array[i-1], 8) == dt
    return dt


def compute_time_step_ratio(time_step, time_step_assimulo):
    assert time_step >= time_step_assimulo
    ratio = time_step / time_step_assimulo
    assert ratio % 1 == 0
    return round(ratio)


def double_check_ratio_errors(t_trapz, t_assimulo, ratio):
    checks = np.random.randint(0, len(t_trapz) - 1, size=15).tolist()
    for i in checks:
        if not round(t_trapz[i], 8) == round(t_assimulo[i * ratio], 8):
            return False
    return True


def errors_analysis(t_sim, studied, t_assimulo, true_states, ratio, error_type='abs'):
    assert double_check_ratio_errors(t_sim, t_assimulo, ratio)
    assert np.array_equal(true_states[0, :], studied[0, :])
    n_checks = studied.shape[0] - 1
    states_to_check = [2, 3, 8, 9, 12, 13, 18, 19, 22, 23, 28, 29]
    errors = np.ones((n_checks, len(states_to_check)))
    for i in range(n_checks):
        for ind, state in enumerate(states_to_check):
            if state in [2, 12, 22]:
                v  = studied[i+1, state]      - studied[i+1, state+7]
                tv = true_states[(i+1)*ratio, state] - true_states[(i+1)*ratio, state+7]
            elif state == 9:
                v  = studied[i+1, 9]      - studied[i+1, 19]
                tv = true_states[(i+1)*ratio, 9] - true_states[(i+1)*ratio, 19]
            elif state == 19:
                v  = studied[i+1, 9]      - studied[i+1, 29]
                tv = true_states[(i+1)*ratio, 9] - true_states[(i+1)*ratio, 29]
            elif state == 29:
                v  = studied[i+1, 19]     - studied[i+1, 29]
                tv = true_states[(i+1)*ratio, 19] - true_states[(i+1)*ratio, 29]
            else:
                v  = studied[i+1, state]
                tv = true_states[(i+1)*ratio, state]
            err = v - tv
            if error_type == 'plot_dif':
                errors[i, ind] = err
            elif error_type == 'abs':
                errors[i, ind] = abs(err)
            elif error_type == 'percentage':
                errors[i, ind] = abs(err / tv) * 100 if tv != 0 else 0.0
    return errors


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser('Plug-n-Play PINN inference (ieee9bus_dae_pinn)')

    p.add_argument('--machine',          type=int,   default=1,
                   choices=[1, 2, 3],    help='Generator to boost with PINN')
    p.add_argument('--pinn_path',        type=str,   default=None,
                   help='Path to trained PINN .pth  (default: plug pretrained model)')
    p.add_argument('--event_type',       type=str,   default='w_setpoint',
                   choices=['w_setpoint', 'p_setpoint'])
    p.add_argument('--event_location',   type=int,   default=3,
                   choices=[1, 2, 3])
    p.add_argument('--event_magnitude',  type=float, default=1e-2)
    p.add_argument('--sim_time',         type=float, default=2.0)
    p.add_argument('--time_step_size',   type=float, default=4e-2)
    p.add_argument('--rk_scheme',        type=str,   default='trapezoidal',
                   choices=['trapezoidal', 'backward_euler'])
    p.add_argument('--study_selection',  type=int,   default=1,
                   choices=[1, 2, 3, 4, 5])
    p.add_argument('--compare_pure_RKscheme',  action='store_true')
    p.add_argument('--compare_ground_truth',   action='store_true', default=True)
    p.add_argument('--gpu',              type=int,   default=0)
    p.add_argument('--save_fig',         action='store_true', default=True)
    p.add_argument('--out_dir',          type=str,   default=None,
                   help='Output directory (default: ./outputs/machine{N})')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    cfg_dir = os.path.join(_HERE, 'config_files')

    # ── choose PINN path ──────────────────────────────────────────────────────
    if args.pinn_path is None:
        # use local pretrained model
        args.pinn_path = os.path.join(_HERE, 'final_models',
                                      f'model_DAE_machine_{args.machine}.pth')
        print(f"Using local pretrained model: {args.pinn_path}")
    else:
        print(f"Using custom PINN: {args.pinn_path}")

    # ── load PINN ─────────────────────────────────────────────────────────────
    simulation_pinn = load_pinn_machine(args.pinn_path, device)
    pinn_ops_limits = compute_pinn_ops_limits(args.pinn_path)

    # ── load system parameters ────────────────────────────────────────────────
    dyn_cfg = config_file(os.path.join(cfg_dir, 'config_machines_dynamic.yaml'))
    sta_cfg = config_file(os.path.join(cfg_dir, 'config_machines_static.yaml'))
    freq, H, Rs, Xd_p, pg_pf, dampings = extract_values_dynamic_components(dyn_cfg)
    volt, Xd, Xq, Xq_p = extract_values_static_components(sta_cfg)
    Yadmittance = torch.load(os.path.join(cfg_dir, 'network_admittance.pt'))

    # verify PINN parameters match config (print warnings for mismatches)
    H_pinn, Xdp_pinn, Pg_pinn, D_pinn = load_pinn_parameters(args.pinn_path)
    m = args.machine - 1
    def _check(name, cfg_val, pinn_val, tol=1e-4):
        if abs(cfg_val - pinn_val) > tol:
            print(f"  Warning: {name} mismatch — config={cfg_val:.6f}  pinn={pinn_val:.6f}")
        else:
            print(f"  {name}: {cfg_val:.6f} ✓")
    print(f"PINN parameter check for machine {args.machine}:")
    _check('H',    H[m],        H_pinn)
    _check('Xd_p', Xd_p[m],    Xdp_pinn)
    _check('Pg',   pg_pf[m],   Pg_pinn)
    _check('D',    dampings[m], D_pinn)

    # ── initial conditions ─────────────────────────────────────────────────────
    gt_file = os.path.join(_HERE, 'gt_simulations',
                           f'sim{int(args.sim_time)}s_{args.event_type}_{args.event_location}.npy')
    has_gt  = os.path.isfile(gt_file)
    if not has_gt:
        args.compare_ground_truth = False
        print(f"Warning: ground truth file not found: {gt_file}")

    ini_cond_sim = load_ini_conditions_true(args, start_value=0)

    # Set output directory based on machine number
    if args.out_dir is None:
        args.out_dir = f'./outputs/machine{args.machine}'
    os.makedirs(args.out_dir, exist_ok=True)

    h    = args.time_step_size
    t_f  = args.sim_time

    # ── helpers ───────────────────────────────────────────────────────────────
    def make_pure_solver(step_size):
        return TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf,
                              ini_cond_sim, t_final=t_f, step_size=step_size)

    def make_hybrid_solver(step_size):
        return TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf,
                              ini_cond_sim, t_final=t_f, step_size=step_size,
                              pinn_boost=args.machine, pinn_weights=simulation_pinn,
                              pinn_limits=pinn_ops_limits)

    # ==========================================================================
    # Study selection  (mirrors plug/main.py exactly)
    # ==========================================================================

    if args.study_selection == 1:
        t_pure, s_pure  = make_pure_solver(h).simulation_main_loop(args.rk_scheme)
        t_pinn, s_pinn  = make_hybrid_solver(h).simulation_main_loop(args.rk_scheme)
        t_true, s_true  = return_true_solution(args)
        plot = trajectories_overview(t_f, t_pinn, s_pinn, t_pure, s_pure, t_true, s_true)
        plot.compute_results(pure_rk_scheme=True, assimulo_states=True)
        fig_name = f'Figure_1_machine{args.machine}'
        plot.show_results(save_fig=args.save_fig,
                          filename=fig_name,
                          output_dir=args.out_dir)

    elif args.study_selection == 2:
        t_pure, s_pure = make_pure_solver(h).simulation_main_loop(args.rk_scheme)
        t_pinn, s_pinn = make_hybrid_solver(h).simulation_main_loop(args.rk_scheme)
        t_true, s_true = return_true_solution(args)
        dt_assimulo    = compute_time_step_assimulo(t_true)
        ratio          = compute_time_step_ratio(h, dt_assimulo)
        err_pure = errors_analysis(t_pure, s_pure, t_true, s_true, ratio)
        err_pinn = errors_analysis(t_pinn, s_pinn, t_true, s_true, ratio)
        plot = custom_overview1(t_f, t_pure, err_pure, t_pinn, err_pinn)
        plot.trajectory_and_errors_plot(8, 10, t_true, s_true, s_pure, s_pinn)
        fig_name = 'Figure_4' if t_f == 10 else 'Figure_5'
        plot.show_results(filename=os.path.join(args.out_dir, fig_name) if args.save_fig else fig_name)
        # 保存仿真结果用于调试
        np.savez(os.path.join(args.out_dir, 'sim_results.npz'),
                 t_pure=t_pure, s_pure=s_pure,
                 t_pinn=t_pinn, s_pinn=s_pinn,
                 t_true=t_true, s_true=s_true)
        print(f"仿真结果已保存到: {os.path.join(args.out_dir, 'sim_results.npz')}")

    elif args.study_selection == 3:
        timesteps = [5e-3, 8e-3, 0.01, 0.02, 0.025, 0.04]
        max_pure   = np.ones((len(timesteps), 2))
        max_hybrid = np.ones((len(timesteps), 2))
        t_true, s_true = return_true_solution(args)
        dt_assimulo    = compute_time_step_assimulo(t_true)
        for i, dt in enumerate(timesteps):
            t_p, s_p = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf,
                                       ini_cond_sim, t_final=t_f, step_size=dt
                                       ).simulation_main_loop(args.rk_scheme)
            t_h, s_h = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf,
                                       ini_cond_sim, t_final=t_f, step_size=dt,
                                       pinn_boost=args.machine,
                                       pinn_weights=simulation_pinn,
                                       pinn_limits=pinn_ops_limits
                                       ).simulation_main_loop(args.rk_scheme)
            ratio = compute_time_step_ratio(dt, dt_assimulo)
            ep = errors_analysis(t_p, s_p, t_true, s_true, ratio)
            eh = errors_analysis(t_h, s_h, t_true, s_true, ratio)
            max_pure[i]   = [np.max(ep[:, 8]), np.max(ep[:, 10])]
            max_hybrid[i] = [np.max(eh[:, 8]), np.max(eh[:, 10])]
            print(f"  step {i+1}/{len(timesteps)}")
        plot = custom_overview2(timesteps)
        plot.show_results(max_pure, max_hybrid)

    elif args.study_selection == 4:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(14, 10))
        gs  = gridspec.GridSpec(1, 2)
        ax0 = plt.subplot(gs[0, 0]); ax0.set_title('Delta_Theta Gen. 3'); ax0.grid()
        ax1 = plt.subplot(gs[0, 1]); ax1.set_title('Voltage magnitude Gen. 3'); ax1.grid()
        t_true_full, s_true_full = return_true_solution_option4(args)
        dt_assimulo = compute_time_step_assimulo(t_true_full)
        ratio = compute_time_step_ratio(h, dt_assimulo)
        starts = np.random.randint(0, len(t_true_full) - 1000, size=30).tolist()
        for k, start in enumerate(starts):
            ic = load_ini_conditions_true_option4(args, start)
            t_p, s_p = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf,
                                       ic, t_final=t_f, step_size=h
                                       ).simulation_main_loop(args.rk_scheme)
            t_h, s_h = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf,
                                       ic, t_final=t_f, step_size=h,
                                       pinn_boost=args.machine,
                                       pinn_weights=simulation_pinn,
                                       pinn_limits=pinn_ops_limits
                                       ).simulation_main_loop(args.rk_scheme)
            end_   = int(start + ratio * t_f / h + 1)
            t_seg, s_seg = return_true_solution_option4(args, start, end_)
            t_seg = t_seg - t_true_full[start]
            ep = errors_analysis(t_p, s_p, t_seg, s_seg, ratio, error_type='plot_dif')
            eh = errors_analysis(t_h, s_h, t_seg, s_seg, ratio, error_type='plot_dif')
            z  = np.zeros((1, ep.shape[1]))
            ax0.plot(t_p, np.vstack([z, ep])[:, 8],  color='orange')
            ax0.plot(t_h, np.vstack([z, eh])[:, 8],  color='blue')
            ax1.plot(t_p, np.vstack([z, ep])[:, 10], color='orange')
            ax1.plot(t_h, np.vstack([z, eh])[:, 10], color='blue')
            print(f"  {k+1}/{len(starts)}")
        plt.tight_layout()
        fname = os.path.join(args.out_dir, 'Figure_7.png')
        plt.savefig(fname, dpi=150); print(f"Saved: {fname}"); plt.close()

    elif args.study_selection == 5:
        timesteps = [0.006, 0.008, 0.01, 0.014, 0.02, 0.024, 0.034, 0.04]
        states_of_interest = [2, 3, 8, 9, 12, 13, 18, 19, 22, 23, 28, 29]
        np.random.seed(25)
        t_true_full, s_true_full = return_true_solution_option4(args)
        dt_assimulo = compute_time_step_assimulo(t_true_full)
        starts = np.random.randint(0, len(t_true_full) - 200, size=100).tolist()
        err_pure_all   = np.ones((len(timesteps), 2 * len(states_of_interest)))
        err_hybrid_all = np.ones((len(timesteps), 2 * len(states_of_interest)))
        for i_dt, dt in enumerate(timesteps):
            print(f"  timestep {dt}")
            ratio = compute_time_step_ratio(dt, dt_assimulo)
            ep_agg = []; eh_agg = []
            for start in starts:
                ic = load_ini_conditions_true_option4(args, start)
                t_p, s_p = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf,
                                           ic, t_final=t_f, step_size=dt
                                           ).simulation_main_loop(args.rk_scheme)
                t_h, s_h = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf,
                                           ic, t_final=t_f, step_size=dt,
                                           pinn_boost=args.machine,
                                           pinn_weights=simulation_pinn,
                                           pinn_limits=pinn_ops_limits
                                           ).simulation_main_loop(args.rk_scheme)
                end_ = int(start + ratio * t_f / dt + 1)
                t_seg, s_seg = return_true_solution_option4(args, start, end_)
                t_seg = t_seg - t_true_full[start]
                ep = errors_analysis(t_p, s_p, t_seg, s_seg, ratio, error_type='percentage')
                eh = errors_analysis(t_h, s_h, t_seg, s_seg, ratio, error_type='percentage')
                ep_agg.append(ep[-1]); eh_agg.append(eh[-1])
            err_pure_all[i_dt]   = np.mean(ep_agg, axis=0)
            err_hybrid_all[i_dt] = np.mean(eh_agg, axis=0)
        np.savez(os.path.join(args.out_dir, 'Figure_8_data.npz'),
                 timesteps=timesteps,
                 err_pure=err_pure_all, err_hybrid=err_hybrid_all)
        print(f"Study 5 data saved to {args.out_dir}/Figure_8_data.npz")


if __name__ == '__main__':
    main()
