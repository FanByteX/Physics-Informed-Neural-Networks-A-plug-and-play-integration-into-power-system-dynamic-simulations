import os
import argparse
import time
import yaml
import numpy as np

import torch

from src.tds_dae_rk_schemes import TDS_simulation

import src.PINN_architecture

from post_processing.trajectories_overview_plot import trajectories_overview
from post_processing.custom_overview_plots import custom_overview1, custom_overview2

parser = argparse.ArgumentParser('TDS-Simulation')
parser.add_argument('--system', type=str, choices=['ieee9bus'], default='ieee9bus')
parser.add_argument('--machine', type=int, choices=[1, 2, 3], default=3)
parser.add_argument('--event_type', choices=['p_setpoint', 'w_setpoint'], default='w_setpoint')
parser.add_argument('--event_location', type=int, choices=[1, 2, 3], default=3)
parser.add_argument('--event_magnitude', type=float, default=1e-2)
parser.add_argument('--sim_time', type=float, default=2.)
parser.add_argument('--time_step_size', type=float, default=4e-2)
parser.add_argument('--rk_scheme', choices=['trapezoidal', 'backward_euler'], default='trapezoidal')
parser.add_argument('--compare_pure_RKscheme', action='store_true')
parser.add_argument('--compare_ground_truth', action='store_true', default=True)
parser.add_argument('--study_selection', type=int, choices=[1, 2, 3, 4, 5, 6], default=2)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


if args.compare_ground_truth:
    gt_dir = './gt_simulations/'
    file_name = f'sim{int(args.sim_time)}s_{args.event_type}_{args.event_location}.npy'
    file_npy = os.path.join(gt_dir, file_name)

    if not os.path.isfile(file_npy):
        args.compare_ground_truth = False

def config_file(yaml_file) -> dict:
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def extract_values_dynamic_components(config_file) -> tuple:
    freq     = config_file['freq']
    H        = list(config_file['inertia_H'].values())
    Rs       = list(config_file['Rs'].values())
    Xd_p     = list(config_file['Xd_prime'].values())
    pg_pf    = list(config_file['Pg_setpoints'].values())
    dampings = list(config_file['Damping_D'].values())
    return freq, H, Rs, Xd_p, pg_pf, dampings

def extract_values_static_components(config_file) -> tuple:
    voltages_magnitude = list(config_file['Voltage_magnitude'].values())
    voltages_angles    = list(config_file['Voltage_angle'].values())
    Xd                 = list(config_file['Xd'].values())
    Xq                 = list(config_file['Xq'].values())
    Xq_prime           = list(config_file['Xq_prime'].values())
    voltages_complex   = np.array(voltages_magnitude)*np.exp(1j*np.array(voltages_angles)*np.pi/180)

    return voltages_complex, Xd, Xq, Xq_prime

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

def load_pinn_machine(pinn_model):
    loaded_pinn = torch.load(pinn_model, map_location=device)
    norm_range, lb_range                     = loaded_pinn['range_norm']
    _, num_neurons, num_layers, inputs, outputs = loaded_pinn['architecture']
    pinn_integrated = src.PINN_architecture.FCN_RESNET(inputs,outputs,num_neurons,num_layers, norm_range, lb_range)
    pinn_integrated.load_state_dict(loaded_pinn['state_dict'])
    pinn_integrated.to(device)
    pinn_integrated.eval()
    return pinn_integrated

def load_pinn_parameters(pinn_model) -> tuple:
    loaded_pinn = torch.load(pinn_model, map_location=device)
    damping_pinn, pg_pinn, H_pinn, Xd_pinn, _ = loaded_pinn['machine_parameters']
    return H_pinn, Xd_pinn, pg_pinn, damping_pinn

def load_ini_conditions_true(start_value):
    gt_dir = './gt_simulations/'
    file_name = f'sim{int(args.sim_time)}s_{args.event_type}_{args.event_location}.npy'
    file_npy = os.path.join(gt_dir, file_name)
    states_assimulo = np.load(file_npy)
    return torch.tensor(states_assimulo[start_value, :-1], dtype=torch.float64)

def load_ini_conditions_true_option4(start_value):
    gt_dir = './gt_simulations/'
    file_name = f'sim10s_{args.event_type}_{args.event_location}.npy'
    file_npy = os.path.join(gt_dir, file_name)
    states_assimulo = np.load(file_npy)
    return torch.tensor(states_assimulo[start_value, :-1], dtype=torch.float64)

def return_true_solution() -> tuple:
    gt_dir = './gt_simulations/'
    file_name = f'sim{int(args.sim_time)}s_{args.event_type}_{args.event_location}.npy'
    file_npy = os.path.join(gt_dir, file_name)
    states_assimulo = np.load(file_npy)
    return states_assimulo[:, 30], states_assimulo[:, :-1]

def return_true_solution_option4(start_point=0, end_point=-1) -> tuple:
    gt_dir = './gt_simulations/'
    file_name = f'sim10s_{args.event_type}_{args.event_location}.npy'
    file_npy = os.path.join(gt_dir, file_name)
    states_assimulo = np.load(file_npy)
    return states_assimulo[start_point:end_point, 30], states_assimulo[start_point:end_point, :-1]

def compute_pinn_ops_limits(pinn_model) -> tuple:
    loaded_pinn = torch.load(pinn_model, map_location=device)
    voltage_limits = loaded_pinn['voltage_stats'][0]
    theta_limits   = loaded_pinn['theta_stats'][0]
    delta_limits   = loaded_pinn['init_state'][2]
    omega_limits   = loaded_pinn['init_state'][3]
    return voltage_limits, theta_limits, delta_limits, omega_limits

def compute_time_step_assimulo(time_array_assimulo):
    time_step_assimulo = round(time_array_assimulo[1] - time_array_assimulo[0], 8)
    for i in np.random.randint(0, len(time_array_assimulo), size=10).tolist():
        assert round(time_array_assimulo[i] - time_array_assimulo[i-1], 8) == time_step_assimulo
    return time_step_assimulo

def compute_time_step_ratio(time_step, time_step_assimulo):
    assert time_step >= time_step_assimulo
    ratio_trapz_assimulo = time_step/time_step_assimulo
    assert ratio_trapz_assimulo % 1 == 0
    ratio_trapz_assimulo = round(ratio_trapz_assimulo)
    return ratio_trapz_assimulo

def double_check_ratio_errors(t_array_trapz, t_array_assimulo, ratio_trapz_assimulo):
    check_time_steps_correct = np.random.randint(0, len(t_array_trapz)-1, size=15).tolist()
    for i in check_time_steps_correct:
        if not round(t_array_trapz[i], 8) == round(t_array_assimulo[i*ratio_trapz_assimulo], 8):
            return False
    return True

def errors_analysis(t_sim_array, studied_states, t_sim_assimulo, states_array_assimulo, ratio_trapz_assimulo, error_type= 'abs'):
    assert double_check_ratio_errors(t_sim_array, t_sim_assimulo, ratio_trapz_assimulo)
    assert np.array_equal(states_array_assimulo[0, :], studied_states[0, :])
    number_of_checks = studied_states.shape[0] - 1
    states_to_check_errors = [2, 3, 8, 9, 12, 13, 18, 19, 22, 23, 28, 29] # state variables and terminal voltages of every machine
    errors_sim_array = np.ones((number_of_checks, len(states_to_check_errors)))
    for i in range(number_of_checks):
        for ind, state in enumerate(states_to_check_errors):
            if state in [2, 12, 22]:
                value_for_error_study      = studied_states[i+1, state] - studied_states[i+1, state+7] 
                true_value_for_error_study = states_array_assimulo[(i+1)*ratio_trapz_assimulo, state] - states_array_assimulo[(i+1)*ratio_trapz_assimulo, state+7]
                current_error  = value_for_error_study - true_value_for_error_study
            elif state == 9:
                value_for_error_study      = studied_states[i+1, state] - studied_states[i+1, 19] 
                true_value_for_error_study = states_array_assimulo[(i+1)*ratio_trapz_assimulo, state] - states_array_assimulo[(i+1)*ratio_trapz_assimulo, 19]
                current_error  = value_for_error_study - true_value_for_error_study
            elif state == 19:
                value_for_error_study      = studied_states[i+1, 9] - studied_states[i+1, 29] 
                true_value_for_error_study = states_array_assimulo[(i+1)*ratio_trapz_assimulo, 9] - states_array_assimulo[(i+1)*ratio_trapz_assimulo, 29]
                current_error  = value_for_error_study - true_value_for_error_study
            elif state == 29:
                value_for_error_study      = studied_states[i+1, 19] - studied_states[i+1, 29] 
                true_value_for_error_study = states_array_assimulo[(i+1)*ratio_trapz_assimulo, 19] - states_array_assimulo[(i+1)*ratio_trapz_assimulo, 29]
                current_error  = value_for_error_study - true_value_for_error_study
            else:
                value_for_error_study      = studied_states[i+1, state]
                true_value_for_error_study = states_array_assimulo[(i+1)*ratio_trapz_assimulo, state]
                current_error  = value_for_error_study - true_value_for_error_study
            if error_type == 'plot_dif':
                errors_sim_array[i, ind] = current_error
            elif error_type == 'abs':
                errors_sim_array[i, ind] = abs(current_error)
            elif error_type == 'percentage':
                errors_sim_array[i, ind] = abs(current_error / true_value_for_error_study) * 100
    return errors_sim_array

if __name__ == "__main__":

    pinn_directory = './final_models/'
    
    # study_selection=6 需要加载所有3个PINN模型
    if args.study_selection == 6:
        simulation_pinn_list = []
        pinn_ops_limits_list = []
        for machine_id in [1, 2, 3]:
            pinn_name = f'model_DAE_machine_{machine_id}.pth'
            pinn_location = os.path.join(pinn_directory, pinn_name)
            simulation_pinn_list.append(load_pinn_machine(pinn_location))
            pinn_ops_limits_list.append(compute_pinn_ops_limits(pinn_location))
        simulation_pinn = simulation_pinn_list
        pinn_ops_limits = pinn_ops_limits_list
    else:
        pinn_name  = f'model_DAE_machine_{args.machine}.pth'
        pinn_location = os.path.join(pinn_directory, pinn_name)
        simulation_pinn = load_pinn_machine(pinn_location)
        pinn_ops_limits = compute_pinn_ops_limits(pinn_location)

    parameters_dc_raw = config_file('./config_files/config_machines_dynamic.yaml')
    freq, H, Rs, Xd_p, pg_pf, dampings = extract_values_dynamic_components(parameters_dc_raw)
    assert len(H)     == 3; assert len(Rs)       == 3; assert len(Xd_p) == 3
    assert len(pg_pf) == 3; assert len(dampings) == 3

    # study_selection=6 时跳过单个机器的参数验证
    if args.study_selection != 6:
        parameters_pinn = load_pinn_parameters(pinn_location)
        assert H[args.machine-1] == parameters_pinn[0]
        assert Xd_p[args.machine-1] == parameters_pinn[1]
        assert pg_pf[args.machine-1] == parameters_pinn[2]
        assert dampings[args.machine-1] == parameters_pinn[3]

    assert all(damp > 0 for damp in dampings)
    assert all(inertia > 0 for inertia in H)

    Yadmittance = torch.load('./config_files/network_admittance.pt')

    parameters_initialization = config_file('./config_files/config_machines_static.yaml')
    volt, Xd, Xq, Xq_p = extract_values_static_components(parameters_initialization)
    ini_cond_sim = load_ini_conditions_true(start_value=0)

    assert args.sim_time > 0.
    assert args.time_step_size > 0.
    t_final_simulations = args.sim_time
    step_size_pure_rk = args.time_step_size
    step_size_hybrid_rk = args.time_step_size

    if args.study_selection == 1:
        solver_pure_rk_method = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=t_final_simulations, step_size=step_size_pure_rk)
        t_test_pure_rk, states_evo = solver_pure_rk_method.simulation_main_loop(integration_scheme=args.rk_scheme)
        solver_hybrid_pinn      = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=t_final_simulations, step_size=step_size_hybrid_rk, 
                                            pinn_boost=args.machine, pinn_weights=simulation_pinn, pinn_limits=pinn_ops_limits)
        t_evo_pinn, states_evo_pinn = solver_hybrid_pinn.simulation_main_loop(integration_scheme=args.rk_scheme)
        t_true, states_true = return_true_solution()
        plotting = trajectories_overview(args.sim_time, t_evo_pinn, states_evo_pinn, t_test_pure_rk, states_evo, t_true, states_true)
        plotting.compute_results(pure_rk_scheme=True, assimulo_states=True)
        plotting.show_results(save_fig=True, filename='Figure_1')
    
    elif args.study_selection == 2:
        solver_pure_rk_method = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=t_final_simulations, step_size=step_size_pure_rk)
        t_test_pure_rk, states_evo = solver_pure_rk_method.simulation_main_loop(integration_scheme=args.rk_scheme)
        solver_hybrid_pinn      = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=t_final_simulations, step_size=step_size_hybrid_rk, 
                                            pinn_boost=args.machine, pinn_weights=simulation_pinn, pinn_limits=pinn_ops_limits)
        t_evo_pinn, states_evo_pinn = solver_hybrid_pinn.simulation_main_loop(integration_scheme=args.rk_scheme)
        t_true, states_true = return_true_solution()
        time_step_assimulo = compute_time_step_assimulo(t_true)
        ratio_trapz_assimulo = compute_time_step_ratio(args.time_step_size, time_step_assimulo)
        errors_pure_array = errors_analysis(t_test_pure_rk, states_evo, t_true, states_true, ratio_trapz_assimulo)
        errors_pinn_array = errors_analysis(t_evo_pinn, states_evo_pinn, t_true, states_true, ratio_trapz_assimulo)
        plotting = custom_overview1(args.sim_time, t_test_pure_rk, errors_pure_array, t_evo_pinn, errors_pinn_array)
        time_step_ms = 8 if args.sim_time == 10 else 40  # Figure_4显示8ms, Figure_5显示40ms
        plotting.trajectory_and_errors_plot(8, 10, t_true, states_true, states_evo, states_evo_pinn, time_step_ms=time_step_ms)
        fig_name = 'Figure_4' if args.sim_time == 10 else 'Figure_5'
        plotting.show_results(filename=fig_name)

    elif args.study_selection == 3:
        timesteps_to_study = [5e-3, 8e-3, 0.01, 0.02, 0.025, 0.04]
        maximums_pure = np.ones((len(timesteps_to_study), 2))
        maximums_hybrid = np.ones((len(timesteps_to_study), 2))
        for i, c_timestep in enumerate(timesteps_to_study):
            solver_pure_rk_method = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=t_final_simulations, step_size=c_timestep)
            t_test_pure_rk, states_evo = solver_pure_rk_method.simulation_main_loop(integration_scheme=args.rk_scheme)
            solver_hybrid_pinn      = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=t_final_simulations, step_size=c_timestep, 
                                                pinn_boost=args.machine, pinn_weights=simulation_pinn, pinn_limits=pinn_ops_limits)
            t_evo_pinn, states_evo_pinn = solver_hybrid_pinn.simulation_main_loop(integration_scheme=args.rk_scheme)
            t_true, states_true = return_true_solution()
            time_step_assimulo = compute_time_step_assimulo(t_true)
            ratio_trapz_assimulo = compute_time_step_ratio(c_timestep, time_step_assimulo)
            errors_pure_array = errors_analysis(t_test_pure_rk, states_evo, t_true, states_true, ratio_trapz_assimulo)
            errors_pinn_array = errors_analysis(t_evo_pinn, states_evo_pinn, t_true, states_true, ratio_trapz_assimulo)
            maximums_pure[i, :] = [np.max(errors_pure_array[:, 8]), np.max(errors_pure_array[:, 10])]
            maximums_hybrid[i, :] = [np.max(errors_pinn_array[:, 8]), np.max(errors_pinn_array[:, 10])]
            print(i+1, len(timesteps_to_study))
        plotting = custom_overview2(timesteps_to_study)
        plotting.show_results(maximums_pure, maximums_hybrid)

    elif args.study_selection == 4:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        
        # 全局字体与样式设置 (强制统一为 Times 风格)
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],  # DejaVu Serif 作为备选
            "mathtext.fontset": "stix",
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 18,
            "axes.unicode_minus": False
        })
        
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(1, 2)
        ax0 = plt.subplot(gs[0, 0])
        ax0.grid()
        ax1 = plt.subplot(gs[0, 1])
        ax1.grid()
        t_true_complete, states_true_complete = return_true_solution_option4()
        time_step_assimulo = compute_time_step_assimulo(t_true_complete)
        ratio_trapz_assimulo = compute_time_step_ratio(step_size_hybrid_rk, time_step_assimulo)
        start_ini_cond = np.random.randint(0, len(t_true_complete)-1000, size=30).tolist()
        for ind, i in enumerate(start_ini_cond):
            ini_cond_sim = load_ini_conditions_true_option4(i)
            solver_pure_rk_method = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=t_final_simulations, step_size=step_size_pure_rk)
            t_test_pure_rk, states_evo = solver_pure_rk_method.simulation_main_loop(integration_scheme=args.rk_scheme)
            solver_hybrid_pinn      = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=t_final_simulations, step_size=step_size_hybrid_rk, 
                                                pinn_boost=args.machine, pinn_weights=simulation_pinn, pinn_limits=pinn_ops_limits)
            t_evo_pinn, states_evo_pinn = solver_hybrid_pinn.simulation_main_loop(integration_scheme=args.rk_scheme)
            t_true_ini, states_true = return_true_solution_option4(i, int(i+ratio_trapz_assimulo*t_final_simulations/step_size_hybrid_rk+1))
            t_true = t_true_ini-t_true_complete[i]
            errors_pure_array = errors_analysis(t_test_pure_rk, states_evo, t_true, states_true, ratio_trapz_assimulo, error_type='plot_dif')
            errors_pinn_array = errors_analysis(t_evo_pinn, states_evo_pinn, t_true, states_true, ratio_trapz_assimulo, error_type='plot_dif')
            zero_initial_errors = np.zeros((1, errors_pure_array.shape[1]))
            errors_simulator_p = np.vstack([zero_initial_errors, errors_pure_array])
            errors_simulator_h = np.vstack([zero_initial_errors, errors_pinn_array])
            ax0.plot(t_test_pure_rk, errors_simulator_p[:, 8], color='b', linestyle='-', linewidth=3, alpha=0.9, label='Pure solver')
            ax0.plot(t_evo_pinn, errors_simulator_h[:, 8], color='r', linestyle='--', linewidth=2.5, alpha=0.9, label='RIA-PINN hybrid solver')
            ax1.plot(t_test_pure_rk, errors_simulator_p[:, 10], color='b', linestyle='-', linewidth=3, alpha=0.9, label='Pure solver')
            ax1.plot(t_evo_pinn, errors_simulator_h[:, 10], color='r', linestyle='--', linewidth=2.5, alpha=0.9, label='RIA-PINN hybrid solver')
            print(ind+1, len(start_ini_cond))
        # 设置y轴标签（加粗）
        ax0.set_ylabel(r"$\boldsymbol{|\delta'_3 - \hat{\delta}'_3|} \ [\mathrm{rad}]$", fontsize=20)
        ax0.set_xlabel(r'$\mathbf{Time\ [s]}$', fontsize=18)
        ax0.tick_params(axis='both', which='major', labelsize=16)
        ax1.set_ylabel(r"$\boldsymbol{|V_3 - \hat{V}_3|} \ [\mathrm{p.u.}]$", fontsize=20)
        ax1.set_xlabel(r'$\mathbf{Time\ [s]}$', fontsize=18)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        
        # 添加统一的图例在顶部居中
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='b', linestyle='-', linewidth=3, label='Pure solver'),
            Line2D([0], [0], color='r', linestyle='--', linewidth=2.5, label='RIA-PINN hybrid solver')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=18, frameon=True, edgecolor='black', fancybox=False)
        
        plt.tight_layout()
        import os
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/Figure_7.png', dpi=150, bbox_inches='tight')
        plt.savefig('outputs/Figure_7.pdf', format='pdf', bbox_inches='tight')
        print('图片已保存：outputs/Figure_7.png')
        print('图片已保存：outputs/Figure_7.pdf')
        plt.close()
    
    elif args.study_selection == 5:
        timesteps_to_study = [0.004, 0.006, 0.008, 0.012, 0.018, 0.020, 0.022, 0.030, 0.032, 0.036, 0.040]
        plotting_states_machine3 = [2, 3, 8, 9, 12, 13, 18, 19, 22, 23, 28, 29]
        np.random.seed(25)
        t_true_complete, states_true_complete = return_true_solution_option4()
        time_step_assimulo = compute_time_step_assimulo(t_true_complete)
        nums_random_true_sim = np.random.randint(0, len(t_true_complete)-200, size=100).tolist()
        
        # 数据文件路径
        import os
        os.makedirs('outputs', exist_ok=True)
        data_file = 'outputs/Figure_9_data.npz'
        
        # 检查是否已有保存的数据
        if os.path.exists(data_file):
            print(f'加载数据文件: {data_file}')
            data = np.load(data_file, allow_pickle=True)
            errors_all_pure = data['errors_all_pure']  # shape: (n_timesteps, n_samples, n_states)
            errors_all_hybrid = data['errors_all_hybrid']
            print('数据加载完成!')
        else:
            print('开始收集误差数据...')
            errors_all_pure = np.ones((len(timesteps_to_study), len(nums_random_true_sim), len(plotting_states_machine3)))
            errors_all_hybrid = np.ones((len(timesteps_to_study), len(nums_random_true_sim), len(plotting_states_machine3)))
            
            for ind_timestep, current_timestep in enumerate(timesteps_to_study):
                print(current_timestep, time_step_assimulo)
                ratio_trapz_assimulo = compute_time_step_ratio(current_timestep, time_step_assimulo)
                for ind_array, num_array_multi in enumerate(nums_random_true_sim):
                    ini_cond_sim = load_ini_conditions_true_option4(num_array_multi)
                    solver_pure_rk_method = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=current_timestep, step_size=current_timestep)
                    t_test_pure_rk, states_evo = solver_pure_rk_method.simulation_main_loop(integration_scheme=args.rk_scheme)
                    solver_hybrid_pinn      = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=current_timestep, step_size=current_timestep, 
                                                        pinn_boost=args.machine, pinn_weights=simulation_pinn, pinn_limits=pinn_ops_limits)
                    t_evo_pinn, states_evo_pinn = solver_hybrid_pinn.simulation_main_loop(integration_scheme=args.rk_scheme)
                    t_true_sim, states_true_sim = return_true_solution_option4(num_array_multi, int(num_array_multi+ratio_trapz_assimulo+1))
                    t_true_sim += -t_true_complete[num_array_multi]
                    errors_pure_array = errors_analysis(t_test_pure_rk, states_evo, t_true_sim, states_true_sim, ratio_trapz_assimulo)
                    errors_pinn_array = errors_analysis(t_evo_pinn, states_evo_pinn, t_true_sim, states_true_sim, ratio_trapz_assimulo)
                    errors_all_pure[ind_timestep, ind_array, :] = errors_pure_array
                    errors_all_hybrid[ind_timestep, ind_array, :] = errors_pinn_array
                print(f'进度: {ind_timestep+1}/{len(timesteps_to_study)}')
            
            # 保存数据
            np.savez(data_file, 
                     errors_all_pure=errors_all_pure, 
                     errors_all_hybrid=errors_all_hybrid,
                     timesteps_to_study=timesteps_to_study,
                     plotting_states_machine3=plotting_states_machine3)
            print(f'数据已保存: {data_file}')
        
        # 计算统计量
        medians_pure = np.median(errors_all_pure, axis=1)
        q1_pure = np.percentile(errors_all_pure, 25, axis=1)
        q3_pure = np.percentile(errors_all_pure, 75, axis=1)
        iqr_pure = q3_pure - q1_pure
        upper_whisker_pure = q3_pure + 1.5 * iqr_pure
        
        medians_hybrid = np.median(errors_all_hybrid, axis=1)
        q1_hybrid = np.percentile(errors_all_hybrid, 25, axis=1)
        q3_hybrid = np.percentile(errors_all_hybrid, 75, axis=1)
        iqr_hybrid = q3_hybrid - q1_hybrid
        upper_whisker_hybrid = q3_hybrid + 1.5 * iqr_hybrid
        
        # ============================================================
        # 绘制折线图 (Figure 9)
        # ============================================================
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.lines import Line2D
        from matplotlib.ticker import LogLocator
        
        # 全局字体与样式设置 (强制统一为 Times 风格)
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],  # DejaVu Serif 作为备选
            "mathtext.fontset": "stix",
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 18,
            "axes.unicode_minus": False
        })
        
        # 原始代码使用plotting_states_final = [8, 10, 4, 6]
        # 这些索引直接对应plotting_states_machine3中的位置
        plotting_states_final = [8, 10]  # 只绘制前两个状态（δ'_3和V_3）
        fig = plt.figure(figsize=(20, 7))  # 与 Figure_4/5 统一尺寸
        gs = gridspec.GridSpec(1, 2)
        axes_list = []
        
        for i, state_idx in enumerate(plotting_states_final):
            ax = plt.subplot(gs[0, i])
            axes_list.append(ax)
            
            # 美化网格线
            ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.3, color='gray')
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2, color='gray')
            
            # 绘制中位数线条，添加标记点
            ax.plot(timesteps_to_study, medians_pure[:, state_idx], 
                    color='b', linestyle='-', linewidth=2.5, alpha=0.9, 
                    marker='o', markersize=7, markerfacecolor='white', markeredgewidth=2,
                    label='Pure solver (median)')
            ax.plot(timesteps_to_study, medians_hybrid[:, state_idx],  
                    color='r', linestyle='-', linewidth=2.5, alpha=0.9,
                    marker='s', markersize=6, markerfacecolor='white', markeredgewidth=2,
                    label='Hybrid solver (median)')
            
            # 绘制上须（虚线），添加阴影区域
            ax.fill_between(timesteps_to_study, 
                            medians_pure[:, state_idx], 
                            upper_whisker_pure[:, state_idx],
                            color='b', alpha=0.15, label='_nolegend_')
            ax.fill_between(timesteps_to_study, 
                            medians_hybrid[:, state_idx], 
                            upper_whisker_hybrid[:, state_idx],
                            color='r', alpha=0.15, label='_nolegend_')
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'Time step $\Delta t$ [s]', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=16)
            
            # 设置x轴显示所有时间步长刻度
            ax.set_xticks(timesteps_to_study)
            ax.set_xticklabels([f'{dt:.3f}' for dt in timesteps_to_study], rotation=60, ha='right', fontsize=11)
            
            # 设置y轴显示更多刻度（对数尺度）
            ax.yaxis.set_major_locator(LogLocator(numticks=15))
            ax.minorticks_on()
            
            # 添加子图标签
            ax.text(0.02, 0.98, f'({chr(97+i)})', transform=ax.transAxes, 
                    fontsize=16, va='top')
        
        # 调整子图底部边距以容纳旋转的x轴标签
        plt.subplots_adjust(bottom=0.2)
        
        # 设置y轴标签
        axes_list[0].set_ylabel(r"$\boldsymbol{|\delta'_3 - \hat{\delta}'_3|} \ [\mathrm{rad}]$", fontsize=20)
        axes_list[1].set_ylabel(r"$\boldsymbol{|V_3 - \hat{V}_3|} \ [\mathrm{p.u.}]$", fontsize=20)
        
        # 添加统一的图例在顶部居中
        legend_elements = [
            Line2D([0], [0], color='b', linestyle='-', linewidth=2.5, marker='o', 
                   markersize=7, markerfacecolor='white', markeredgewidth=2, label='Pure solver'),
            Line2D([0], [0], color='b', linestyle='-', alpha=0.2, linewidth=8, label='IQR range'),
            Line2D([0], [0], color='r', linestyle='-', linewidth=2.5, marker='s', 
                   markersize=6, markerfacecolor='white', markeredgewidth=2, label='RIA-PINN hybrid solver'),
            Line2D([0], [0], color='r', linestyle='-', alpha=0.2, linewidth=8, label='IQR range')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=4, fontsize=18, frameon=True, edgecolor='black', fancybox=False)
        
        plt.tight_layout()
        plt.savefig('outputs/Figure_9.png', dpi=150, bbox_inches='tight')
        plt.savefig('outputs/Figure_9.pdf', format='pdf', bbox_inches='tight')
        print('折线图已保存：outputs/Figure_9.png')
        plt.close()
        
        # ============================================================
        # 绘制箱线图 (Figure 9 boxplot)
        # ============================================================
        print('正在生成箱线图...')
        fig2 = plt.figure(figsize=(16, 6))
        gs2 = gridspec.GridSpec(1, 2)
        
        for i, state_idx in enumerate(plotting_states_final):
            ax = plt.subplot(gs2[0, i])
            ax.grid(axis='y', alpha=0.3)
            
            if i == 0:
                title = r"$\delta'_3$ Prediction Error"
            else:
                title = r"$V_3$ Prediction Error"
            
            # 准备箱线图数据 - 每个时间步有两个箱线（Pure和Hybrid并排）
            positions = []
            box_data = []
            colors = []
            
            for j in range(len(timesteps_to_study)):
                # Pure solver box
                positions.append(j * 3)
                box_data.append(errors_all_pure[j, :, state_idx])
                colors.append('steelblue')
                # Hybrid solver box
                positions.append(j * 3 + 1)
                box_data.append(errors_all_hybrid[j, :, state_idx])
                colors.append('indianred')
            
            # 绘制箱线图
            bp = ax.boxplot(box_data, positions=positions, widths=0.8, patch_artist=True,
                            showfliers=False,
                            medianprops=dict(color='black', linewidth=1.5),
                            whiskerprops=dict(linewidth=1),
                            capprops=dict(linewidth=1))
            
            # 设置箱体颜色
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_yscale('log')
            ax.set_xlabel(r'$\mathbf{\Delta t\ [s]}$', fontsize=18)
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=14)
            
            # 设置x轴刻度标签（显示在每对箱线图中间）
            tick_positions = [j * 3 + 0.5 for j in range(len(timesteps_to_study))]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f'{dt:.3f}' for dt in timesteps_to_study], rotation=60, ha='right', fontsize=10)
            ax.set_xlim(-1, len(timesteps_to_study) * 3 - 0.5)
        
        # 设置y轴标签
        axes_box = fig2.get_axes()
        axes_box[0].set_ylabel(r"$\boldsymbol{|\delta'_3 - \hat{\delta}'_3|} \ [\mathrm{rad}]$", fontsize=18)
        axes_box[1].set_ylabel(r"$\boldsymbol{|V_3 - \hat{V}_3|} \ [\mathrm{p.u.}]$", fontsize=18)
        
        # 添加图例
        legend_elements_box = [
            Line2D([0], [0], color='steelblue', linewidth=10, alpha=0.7, label='Pure solver'),
            Line2D([0], [0], color='indianred', linewidth=10, alpha=0.7, label='Hybrid solver')
        ]
        fig2.legend(handles=legend_elements_box, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
                    ncol=2, fontsize=14, frameon=True, edgecolor='black', fancybox=False)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2, top=0.88)
        plt.savefig('outputs/Figure_9_boxplot.png', dpi=150, bbox_inches='tight')
        plt.savefig('outputs/Figure_9_boxplot.pdf', format='pdf', bbox_inches='tight')
        print('箱线图已保存：outputs/Figure_9_boxplot.png')
        plt.close()
    
    elif args.study_selection == 6:
        
        # 添加图例
        legend_elements_box = [
            Line2D([0], [0], color='steelblue', linewidth=10, alpha=0.7, label='Pure solver'),
            Line2D([0], [0], color='indianred', linewidth=10, alpha=0.7, label='Hybrid solver')
        ]
        fig2.legend(handles=legend_elements_box, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
                    ncol=2, fontsize=14, frameon=True, edgecolor='black', fancybox=False)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2, top=0.88)
        plt.savefig('outputs/Figure_9_boxplot.png', dpi=150, bbox_inches='tight')
        plt.savefig('outputs/Figure_9_boxplot.pdf', format='pdf', bbox_inches='tight')
        print('箱线图已保存：outputs/Figure_9_boxplot.png')
        print('箱线图已保存：outputs/Figure_9_boxplot.pdf')
        plt.close()
    
    elif args.study_selection == 6:
        # 同时加载3个PINN模型，对3台机器分别进行加速
        solver_pure_rk_method = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=t_final_simulations, step_size=step_size_pure_rk)
        t_test_pure_rk, states_evo = solver_pure_rk_method.simulation_main_loop(integration_scheme=args.rk_scheme)
        solver_hybrid_pinn = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=t_final_simulations, step_size=step_size_hybrid_rk, 
                                            pinn_boost='all', pinn_weights=simulation_pinn, pinn_limits=pinn_ops_limits)
        t_evo_pinn, states_evo_pinn = solver_hybrid_pinn.simulation_main_loop(integration_scheme=args.rk_scheme)
        t_true, states_true = return_true_solution()
        plotting = trajectories_overview(args.sim_time, t_evo_pinn, states_evo_pinn, t_test_pure_rk, states_evo, t_true, states_true)
        plotting.compute_results(pure_rk_scheme=True, assimulo_states=True)
        plotting.show_results(save_fig=True, filename='Figure_9')