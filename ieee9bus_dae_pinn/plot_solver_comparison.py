"""
plot_solver_comparison.py
=========================
Plot Pure RK solver, PINN-boosted solver, and ground truth comparison.

Usage:
    python plot_solver_comparison.py --machine 1 --sim_time 10 --time_step_size 8e-3 --pinn_path ./final_models/model_DAE_machine_1_trained.pth
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'src'))

from tds_dae_rk_schemes import TDS_simulation
from train_plug_pinn import FCN_RESNET, FCN_simple


class PINNWrapper(torch.nn.Module):
    """包装FCN模型以兼容plug的pinn_integration_scheme"""
    def __init__(self, base_model, n_input):
        super().__init__()
        self.base = base_model
        self.n_in = n_input

    def forward(self, x):
        dtype = next(self.base.parameters()).dtype
        x = x.to(dtype)
        # plug的7维模型需要复制第4维(omicron_0)到第5维
        # 输入格式: [Vm_0, Vm, theta_pend, omicron_0, omega_0, h]
        # 模型期望: [Vm_0, Vm, theta_pend, omicron_0, delta_0, omega_0, h]
        if self.n_in == 7 and x.shape[-1] == 6:
            # 在位置4插入复制的omicron_0作为delta_0近似
            x = torch.cat([x[..., :4], x[..., 3:4], x[..., 4:]], dim=-1)
        return self.base(x)


def load_pinn_model(pinn_path, device='cpu'):
    """从checkpoint加载PINN模型"""
    ckpt = torch.load(pinn_path, map_location=device)
    norm_range, lb_range = ckpt['range_norm']
    arch = ckpt['architecture']
    
    arch_type = ckpt.get('architecture_type', None)
    
    if len(arch) == 5:
        N_HIDDEN, N_HIDDEN_RES, N_LAYERS, N_INPUT, N_OUTPUT = arch
        if arch_type is None:
            arch_type = 'resnet'
        print(f"Loading ResNet model: hidden={N_HIDDEN}, layers={N_LAYERS}, input={N_INPUT}")
        model = FCN_RESNET(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, norm_range, lb_range)
    elif len(arch) == 4:
        N_HIDDEN, N_LAYERS, N_INPUT, N_OUTPUT = arch
        if arch_type is None:
            arch_type = 'fcn'
        print(f"Loading FCN_simple model: hidden={N_HIDDEN}, layers={N_LAYERS}, input={N_INPUT}")
        model = FCN_simple(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, norm_range, lb_range)
        # 确保buffer已设置
        if not hasattr(model, 'range_states') or model.range_states is None:
            model.register_buffer('range_states', torch.tensor(norm_range, dtype=torch.float32))
        if not hasattr(model, 'lb_states') or model.lb_states is None:
            model.register_buffer('lb_states', torch.tensor(lb_range, dtype=torch.float32))
    else:
        raise ValueError(f"Unknown architecture format: {arch}")
    
    # 加载权重，忽略已存在的buffer
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    wrapped = PINNWrapper(model, n_input=N_INPUT)
    wrapped.to(device)
    wrapped.eval()
    return wrapped


def get_state_indices(machine_id):
    """
    状态向量结构 (30维):
    [Eq', Ed', delta, omega, Id, Iq, Id_g, Iq_g, Vm, Theta] × 3 machines
    
    machine_id: 1, 2, or 3
    返回该发电机的关键状态索引
    """
    m = machine_id - 1  # 0-based index
    return {
        'Eq_prime': m * 10 + 0,
        'Ed_prime': m * 10 + 1,
        'delta': m * 10 + 2,
        'omega': m * 10 + 3,
        'Id': m * 10 + 4,
        'Iq': m * 10 + 5,
        'Id_g': m * 10 + 6,
        'Iq_g': m * 10 + 7,
        'Vm': m * 10 + 8,
        'Theta': m * 10 + 9,
    }


def config_file(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def extract_values_dynamic_components(cfg):
    """提取动态参数"""
    freq = cfg.get('freq', 60.0)
    H = [cfg['inertia_H']['machine1'], cfg['inertia_H']['machine2'], cfg['inertia_H']['machine3']]
    Rs = [cfg['Rs']['machine1'], cfg['Rs']['machine2'], cfg['Rs']['machine3']]
    Xd_p = [cfg['Xd_prime']['machine1'], cfg['Xd_prime']['machine2'], cfg['Xd_prime']['machine3']]
    pg_pf = [cfg['Pg_setpoints']['machine1'], cfg['Pg_setpoints']['machine2'], cfg['Pg_setpoints']['machine3']]
    dampings = [cfg['Damping_D']['machine1'], cfg['Damping_D']['machine2'], cfg['Damping_D']['machine3']]
    return freq, H, Rs, Xd_p, pg_pf, dampings


def load_ini_conditions_true(args, start_value=0):
    """加载初始条件 - 数据格式: (N, 31) 最后列是时间"""
    gt_file = os.path.join(_HERE, 'gt_simulations', 
                           f'sim{int(args.sim_time)}s_{args.event_type}_{args.event_location}.npy')
    if os.path.exists(gt_file):
        data = np.load(gt_file)  # shape: (N, 31)
        states = data[:, :30]    # 前30列是状态
        return torch.tensor(states[start_value], dtype=torch.float64)
    else:
        # 默认初始条件
        return torch.zeros(30, dtype=torch.float64)


def return_true_solution(args):
    """返回真实解 - 数据格式: (N, 31) 最后列是时间"""
    gt_file = os.path.join(_HERE, 'gt_simulations',
                           f'sim{int(args.sim_time)}s_{args.event_type}_{args.event_location}.npy')
    if os.path.exists(gt_file):
        data = np.load(gt_file)  # shape: (N, 31)
        states = data[:, :30]    # 前30列是状态
        t = data[:, 30]          # 最后一列是时间
        return t, states
    return None, None


def main():
    parser = argparse.ArgumentParser(description='Plot Pure RK, PINN-boosted solver, and ground truth comparison')
    parser.add_argument('--machine', type=int, default=1, choices=[1,2,3],
                        help='Generator to analyze')
    parser.add_argument('--sim_time', type=float, default=10,
                        help='Simulation time in seconds')
    parser.add_argument('--time_step_size', type=float, default=8e-3,
                        help='Time step size')
    parser.add_argument('--rk_scheme', type=str, default='trapezoidal',
                        choices=['trapezoidal', 'backward_euler'])
    parser.add_argument('--event_type', type=str, default='w_setpoint')
    parser.add_argument('--event_location', type=int, default=3)
    parser.add_argument('--pinn_path', type=str, default=None,
                        help='Path to trained PINN model (optional)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory')
    args = parser.parse_args()

    # 输出目录
    out_dir = args.out_dir or os.path.join(_HERE, 'outputs', f'machine{args.machine}')
    os.makedirs(out_dir, exist_ok=True)

    # 获取状态索引
    idx = get_state_indices(args.machine)
    print(f"Machine {args.machine} 状态索引:")
    for name, i in idx.items():
        print(f"  {name}: {i}")

    # 加载配置
    cfg_dir = os.path.join(_HERE, 'config_files')
    dyn_cfg = config_file(os.path.join(cfg_dir, 'config_machines_dynamic.yaml'))
    freq, H, Rs, Xd_p, pg_pf, dampings = extract_values_dynamic_components(dyn_cfg)
    Yadmittance = torch.load(os.path.join(cfg_dir, 'network_admittance.pt'))
    
    # 初始条件
    ini_cond = load_ini_conditions_true(args, start_value=0)
    
    # ========== 纯 RK 求解器 ==========
    print(f"\n[1/2] 运行纯 {args.rk_scheme} 求解器...")
    solver_pure = TDS_simulation(
        dampings, freq, H, Xd_p, Yadmittance, pg_pf,
        ini_cond, t_final=args.sim_time, step_size=args.time_step_size,
        pinn_boost=None, pinn_weights=None, pinn_limits=None
    )
    t_pure, states_pure = solver_pure.simulation_main_loop(args.rk_scheme)
    print(f"纯求解器完成: {len(t_pure)} 时间步")

    # ========== PINN-boosted 求解器 ==========
    states_pinn = None
    t_pinn = None
    if args.pinn_path and os.path.exists(args.pinn_path):
        print(f"\n[2/2] 加载PINN模型: {args.pinn_path}")
        pinn = load_pinn_model(args.pinn_path, device='cpu')
        
        # 设置PINN limits: [limits_v, limits_theta, limits_delta, limits_omega]
        # 基于典型的电力系统运行范围 (扩大范围避免警告)
        pinn_limits = [
            torch.tensor([0.5, 1.5], dtype=torch.float64),    # Vm 电压范围
            torch.tensor([-10.0, 10.0], dtype=torch.float64), # Theta 相角范围
            torch.tensor([-10.0, 10.0], dtype=torch.float64), # delta 范围
            torch.tensor([-1.0, 1.0], dtype=torch.float64),   # omega 偏差范围
        ]
        
        print(f"运行 PINN-boosted {args.rk_scheme} 求解器 (machine {args.machine})...")
        solver_pinn = TDS_simulation(
            dampings, freq, H, Xd_p, Yadmittance, pg_pf,
            ini_cond, t_final=args.sim_time, step_size=args.time_step_size,
            pinn_boost=args.machine, pinn_weights=pinn, pinn_limits=pinn_limits
        )
        t_pinn, states_pinn = solver_pinn.simulation_main_loop(args.rk_scheme)
        print(f"PINN求解器完成: {len(t_pinn)} 时间步")
    else:
        print(f"\n[2/2] 未指定PINN模型，跳过PINN-boosted求解器")

    # ========== 加载 ground truth ==========
    t_true, states_true = return_true_solution(args)
    if t_true is not None:
        print(f"Ground truth 加载完成: {len(t_true)} 时间步")
    else:
        print("未找到 ground truth")

    # ========== 绘图 ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    plot_states = [
        ('delta', 'Delta [rad]', idx['delta']),
        ('omega', 'Omega [p.u.]', idx['omega']),
        ('Vm', 'Voltage Magnitude [p.u.]', idx['Vm']),
        ('Theta', 'Theta [rad]', idx['Theta']),
    ]

    for ax, (name, ylabel, state_idx) in zip(axes.flatten(), plot_states):
        # Ground truth (黑色虚线)
        if t_true is not None and states_true is not None:
            ax.plot(t_true, states_true[:, state_idx], 'k--', linewidth=1.5, label='Ground truth', zorder=10)
        
        # 纯求解器 (蓝色实线)
        ax.plot(t_pure, states_pure[:, state_idx], 'b-', linewidth=1.2, label='Pure RK solver', alpha=0.8)
        
        # PINN求解器 (红色实线)
        if t_pinn is not None and states_pinn is not None:
            ax.plot(t_pinn, states_pinn[:, state_idx], 'r-', linewidth=1.2, label='PINN-boosted solver', alpha=0.8)
        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{name} - Gen.{args.machine}')
        ax.grid(True, alpha=0.3)
        ax.legend()

    title = f'Solver Comparison (Machine {args.machine}, dt={args.time_step_size})'
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()

    fig_path = os.path.join(out_dir, f'solver_comparison_machine{args.machine}.png')
    plt.savefig(fig_path, dpi=150)
    print(f"\n图片已保存: {fig_path}")
    plt.close()

    # ========== 误差统计 ==========
    if t_true is not None and states_true is not None:
        print("\n" + "="*60)
        print("误差统计 (vs Ground Truth)")
        print("="*60)
        
        print(f"\n{'状态':<10} {'纯求解器 Max':>15} {'纯求解器 Mean':>15}", end='')
        if states_pinn is not None:
            print(f" {'PINN求解器 Max':>15} {'PINN求解器 Mean':>15}")
        else:
            print()
        
        for name, ylabel, state_idx in plot_states:
            # 纯求解器误差
            states_true_interp_pure = np.interp(t_pure, t_true, states_true[:, state_idx])
            error_pure = np.abs(states_pure[:, state_idx] - states_true_interp_pure)
            
            print(f"{name:<10} {error_pure.max():>15.6f} {error_pure.mean():>15.6f}", end='')
            
            # PINN求解器误差
            if states_pinn is not None:
                states_true_interp_pinn = np.interp(t_pinn, t_true, states_true[:, state_idx])
                error_pinn = np.abs(states_pinn[:, state_idx] - states_true_interp_pinn)
                print(f" {error_pinn.max():>15.6f} {error_pinn.mean():>15.6f}")
            else:
                print()


if __name__ == '__main__':
    main()
