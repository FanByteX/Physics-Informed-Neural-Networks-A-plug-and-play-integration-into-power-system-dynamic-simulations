"""检查推理误差"""
import os
import sys
import yaml
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'src'))

from tds_dae_rk_schemes import TDS_simulation
from train_plug_pinn import FCN_simple

# 加载PINN
pinn_path = './logs/plug_pinn/20260404_002521/model_best.pth'
ckpt = torch.load(pinn_path, map_location='cpu')
norm_range, lb_range = ckpt['range_norm']
arch = ckpt['architecture']
N_HIDDEN, N_LAYERS, N_INPUT, N_OUTPUT = arch
model = FCN_simple(N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, norm_range, lb_range)
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval()

class PINNWrapper(torch.nn.Module):
    def __init__(self, base_model, n_input):
        super().__init__()
        self.base = base_model
        self.n_in = n_input
    def forward(self, x):
        dtype = next(self.base.parameters()).dtype
        x = x.to(dtype)
        if x.shape[-1] == self.n_in - 1:
            x = torch.cat([x[..., :4], x[..., 3:4], x[..., 4:]], dim=-1)
        return self.base(x)

simulation_pinn = PINNWrapper(model, n_input=N_INPUT)

# 加载配置
cfg_dir = os.path.join(_PLUG, 'config_files')
dyn_cfg = yaml.safe_load(open(os.path.join(cfg_dir, 'config_machines_dynamic.yaml')))
sta_cfg = yaml.safe_load(open(os.path.join(cfg_dir, 'config_machines_static.yaml')))

freq = dyn_cfg['freq']
H = list(dyn_cfg['inertia_H'].values())
Rs = list(dyn_cfg['Rs'].values())
Xd_p = list(dyn_cfg['Xd_prime'].values())
pg_pf = list(dyn_cfg['Pg_setpoints'].values())
dampings = list(dyn_cfg['Damping_D'].values())

volt_mag = list(sta_cfg['Voltage_magnitude'].values())
volt_ang = list(sta_cfg['Voltage_angle'].values())
volt = np.array(volt_mag) * np.exp(1j * np.array(volt_ang) * np.pi / 180)
Yadmittance = torch.load(os.path.join(cfg_dir, 'network_admittance.pt'))

# 初始条件
gt_file = os.path.join(_PLUG, 'gt_simulations', 'sim10s_w_setpoint_3.npy')
states_all = np.load(gt_file)
ini_cond = torch.tensor(states_all[0, :-1], dtype=torch.float64)

# 运行仿真
h = 8e-3
t_f = 10.0
machine = 1

# 纯求解器
print("运行纯求解器...")
pure_solver = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf,
                              ini_cond, t_final=t_f, step_size=h)
t_pure, s_pure = pure_solver.simulation_main_loop('trapezoidal')

# PINN增强求解器
print("运行PINN增强求解器...")
pinn_limits = ckpt['voltage_stats'][0], ckpt['theta_stats'][0], ckpt['init_state'][2], ckpt['init_state'][3]
hybrid_solver = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf,
                                ini_cond, t_final=t_f, step_size=h,
                                pinn_boost=machine, pinn_weights=simulation_pinn,
                                pinn_limits=pinn_limits)
t_pinn, s_pinn = hybrid_solver.simulation_main_loop('trapezoidal')

# 真实值
t_true = states_all[:, 30]
s_true = states_all[:, :-1]

print()
print('=== 仿真结果检查 ===')
print(f't_pure shape: {t_pure.shape}, s_pure shape: {s_pure.shape}')
print(f't_pinn shape: {t_pinn.shape}, s_pinn shape: {s_pinn.shape}')
print(f't_true shape: {t_true.shape}, s_true shape: {s_true.shape}')

# 计算真实时间步长比例
dt_assimulo = round(t_true[1] - t_true[0], 8)
ratio = round(h / dt_assimulo)
print(f'dt_assimulo: {dt_assimulo}, ratio: {ratio}')

# 手动计算误差
states_to_check = [2, 3, 8, 9, 12, 13, 18, 19, 22, 23, 28, 29]

def compute_errors(t_sim, studied, t_true, true_states, ratio):
    n_checks = studied.shape[0] - 1
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
            errors[i, ind] = abs(v - tv)
    return errors

err_pure = compute_errors(t_pure, s_pure, t_true, s_true, ratio)
err_pinn = compute_errors(t_pinn, s_pinn, t_true, s_true, ratio)

print()
print('=== 误差分析 ===')
print('states_to_check:', states_to_check)
print('  索引 8 -> state 22 = Delta_Theta Gen3 (实际是 delta - theta)')
print('  索引 10 -> state 28 = Vm Gen3')
print()
print(f'纯求解器 Delta_Theta 最大误差: {np.max(err_pure[:, 8]):.6f}')
print(f'PINN求解器 Delta_Theta 最大误差: {np.max(err_pinn[:, 8]):.6f}')
print(f'纯求解器 Vm 最大误差: {np.max(err_pure[:, 10]):.6f}')
print(f'PINN求解器 Vm 最大误差: {np.max(err_pinn[:, 10]):.6f}')

print()
print('=== 轨迹对比 (多个时间点) ===')
for t_check in [2.0, 5.0, 8.0]:
    idx = int(t_check / h)
    true_idx = idx * ratio
    print(f'\n--- t = {t_check}s ---')
    print(f'Delta_Theta (Gen3) [state_1=8 -> state 22]:')
    print(f'  真实值: {s_true[true_idx, 22] - s_true[true_idx, 29]:.6f}')
    print(f'  纯求解器: {s_pure[idx, 22] - s_pure[idx, 29]:.6f}, 误差: {err_pure[idx, 8]:.6f}')
    print(f'  PINN求解器: {s_pinn[idx, 22] - s_pinn[idx, 29]:.6f}, 误差: {err_pinn[idx, 8]:.6f}')
    print(f'Vm (Gen3) [state_2=10 -> state 28]:')
    print(f'  真实值: {s_true[true_idx, 28]:.6f}')
    print(f'  纯求解器: {s_pure[idx, 28]:.6f}, 误差: {err_pure[idx, 10]:.6f}')
    print(f'  PINN求解器: {s_pinn[idx, 28]:.6f}, 误差: {err_pinn[idx, 10]:.6f}')
