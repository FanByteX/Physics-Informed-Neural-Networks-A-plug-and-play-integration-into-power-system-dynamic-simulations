#!/usr/bin/env python
import numpy as np
from post_processing.custom_overview_plots import custom_overview1

# Load ground truth data
gt_data = np.load('gt_simulations/sim2s_w_setpoint_3.npy', allow_pickle=True).item()
t_true = gt_data['t']
states_true = gt_data['states']

# Load simulation results for pure solver (timestep 0.02s)
output_data_pure = np.load('outputs/machine1_test/states_evo_pure_0.02.npy', allow_pickle=True).item()
t_pure = output_data_pure['t']
states_pure = output_data_pure['states']

# Load simulation results for PINN hybrid (timestep 0.02s)
output_data_hybrid = np.load('outputs/machine1_test/states_evo_hybrid_0.02.npy', allow_pickle=True).item()
t_hybrid = output_data_hybrid['t']
states_hybrid = output_data_hybrid['states']

# Compute errors
def errors_analysis(t_test, states_test, t_true, states_true, ratio=1.0):
    """Compute errors between test and true trajectories"""
    states_true_interp = np.zeros((len(t_test), states_true.shape[1]))
    for i in range(states_true.shape[1]):
        states_true_interp[:, i] = np.interp(t_test, t_true, states_true[:, i])
    
    errors = np.abs(states_test - states_true_interp)
    return errors

# Compute time step ratio (assuming ground truth is at 0.005s)
dt_true = t_true[1] - t_true[0]
dt_test = t_pure[1] - t_pure[0]
ratio = dt_test / dt_true

errors_pure = errors_analysis(t_pure, states_pure, t_true, states_true, ratio)
errors_hybrid = errors_analysis(t_hybrid, states_hybrid, t_true, states_true, ratio)

# Create plot
plotting = custom_overview1(2.0, t_pure, errors_pure, t_hybrid, errors_hybrid)
plotting.trajectory_and_errors_plot(8, 10, t_true, states_true, states_pure, states_hybrid)

# Save as Figure_5
plotting.show_results(filename='Figure_5')
print('Figure_5.png generated successfully')
