"""
RES-PINN-DAE 模块
基于MindSpore的残差物理信息神经网络求解刚性DAE系统
"""
__version__ = "1.0.0"
__author__ = "RES-PINN-DAE Team"

from .models.networks import fnn, attention, Conv1D
from .models.dae import three_bus_PN, power_net_dae
from .training.trainer import supervisor, State, LossHistory, ModelCheckPoint
from .training.irk import get_irk_weights_times
from .data.dataset import dae_data, MSE
from .utils.common import timing, dotdict, l2_relative_error, training_display
from .utils.plots import plot_loss, plot_trajectories, plot_l2_errors, save_prediction_data
from .config import get_config, update_config

__all__ = [
    # Networks
    'fnn', 'attention', 'Conv1D',
    # DAE Models
    'three_bus_PN', 'power_net_dae',
    # Training
    'supervisor', 'State', 'LossHistory', 'ModelCheckPoint',
    # IRK
    'get_irk_weights_times',
    # Data
    'dae_data', 'MSE',
    # Utils
    'timing', 'dotdict', 'l2_relative_error', 'training_display',
    # Plots
    'plot_loss', 'plot_trajectories', 'plot_l2_errors', 'save_prediction_data',
    # Config
    'get_config', 'update_config',
]
