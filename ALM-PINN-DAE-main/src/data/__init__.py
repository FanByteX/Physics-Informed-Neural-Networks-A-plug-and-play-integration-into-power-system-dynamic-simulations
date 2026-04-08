"""
数据模块
"""
from .base import Data
from .irk_weights import get_irk_weights_times
from .dae_data import dae_data

__all__ = ['Data', 'get_irk_weights_times', 'dae_data']
