import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# 全局字体与样式设置 (强制统一为 Times 风格)
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],  # DejaVu Serif 作为备选
    "mathtext.fontset": "stix",
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 18,
    "axes.unicode_minus": False
})

class custom_overview1:
    def __init__(self, sim_time, t_test_solver1, errors_solver1, t_test_solver2, errors_solver2) -> None:
        assert sim_time > 0
        assert t_test_solver1[-1] == sim_time
        assert t_test_solver2[-1] == sim_time
        self.simulation_range = sim_time
        self.t_test_solver1 = t_test_solver1
        self.t_test_solver2 = t_test_solver2
        self.errors_solver1 = errors_solver1
        self.errors_solver2 = errors_solver2
        self.num_plots = errors_solver1.shape[1]
        if len(t_test_solver1) > errors_solver1.shape[0]:
            self.errors_solver1 = self.add_zeros_initial_value(errors_solver1)
        if len(t_test_solver2) > errors_solver2.shape[0]:
            self.errors_solver2 = self.add_zeros_initial_value(errors_solver2)
        assert len(t_test_solver1) == self.errors_solver1.shape[0]
        assert len(t_test_solver2) == self.errors_solver2.shape[0]

    def trajectory_and_errors_plot(self, state_1, state_2, time_array_true, states_array_true, states_array_pure, states_array_hybrid, time_step_ms=40):
        assert self.errors_solver1.shape[1] == self.errors_solver2.shape[1]
        fig = plt.figure(figsize=(20, 7))  # 放大尺寸以适合期刊模板
        gs = gridspec.GridSpec(1, 2)
        ax0 = plt.subplot(gs[0, 0])
        #ax0.set_title(r'$\delta - \theta$ Gen. 3', fontsize=20, fontweight='bold')
        ax0.grid()
        ax1 = plt.subplot(gs[0, 1])
        #ax1.set_title(r'$V_m$ Gen. 3', fontsize=20, fontweight='bold')
        ax1.grid()

        # 先画 Pure solver 和 PINN hybrid，再画 Ground truth 在最上层
        ax0.plot(self.t_test_solver1, states_array_pure[:, 22]- states_array_pure[:, 29], color='b', linestyle='-', linewidth=3, alpha=0.9, label='Pure solver') # Pure solver: 蓝色实线
        ax0.plot(self.t_test_solver2, states_array_hybrid[:, 22]- states_array_hybrid[:, 29], color='r', linestyle='--', linewidth=2.5, alpha=0.9, label='RIA-PINN hybrid solver') # PINN hybrid: 红色虚线
        ax0.plot(time_array_true, states_array_true[:, 22]-states_array_true[:, 29], color='k', linestyle='--', label='Ground truth') # Ground truth: 黑色虚线在最上层
        #ax0.set_ylabel(r"$\delta'_3 = \delta_3 - \theta_3 \ [\mathrm{rad}]$", fontsize=20)  #不加粗
        ax0.set_ylabel(r"$\boldsymbol{\delta'_3 = \delta_3 - \theta_3} \ [\mathrm{rad}]$", fontsize=20)  #加粗变量
        ax0.set_xlabel(r'$\mathbf{Time\ [s]}$', fontsize=18)
        ax0.tick_params(axis='both', which='major', labelsize=16)

        # 先画 Pure solver 和 PINN hybrid，再画 Ground truth 在最上层
        ax1.plot(self.t_test_solver1, states_array_pure[:, 28], color='b', linestyle='-', linewidth=3, alpha=0.9)
        ax1.plot(self.t_test_solver2, states_array_hybrid[:, 28], color='r', linestyle='--', linewidth=2.5, alpha=0.9)
        ax1.plot(time_array_true, states_array_true[:, 28], color='k', linestyle='--')  # Ground truth: 黑色虚线在最上层
        ax1.set_ylabel(r"$V_3 \ [\mathrm{p.u.}]$", fontsize=20)
        ax1.set_xlabel(r'$\mathbf{Time\ [s]}$', fontsize=18)
        ax1.tick_params(axis='both', which='major', labelsize=16)

        ax0_twin = ax0.twinx()
        ax0_twin.set_ylabel(r"$\boldsymbol{|\delta'_3 - \hat{\delta}'_3|} \ [\mathrm{rad}]$", fontsize=20, labelpad=1)
        ax0_twin.tick_params(axis='both', which='major', labelsize=16)
        ax1_twin = ax1.twinx()
        ax1_twin.set_ylabel(r"$\boldsymbol{|V_3 - \hat{V}_3|} \ [\mathrm{p.u.}]$", fontsize=20, fontweight='bold')
        ax1_twin.tick_params(axis='both', which='major', labelsize=16)

        ax0_twin.plot(self.t_test_solver1, self.errors_solver1[:, state_1], color='b', linestyle='-', alpha=0.6)
        ax0_twin.fill_between(self.t_test_solver1, self.errors_solver1[:, state_1], color='b', alpha=0.3)
        ax0_twin.plot(self.t_test_solver2, self.errors_solver2[:, state_1], color='r', linestyle='--', alpha=0.6)
        ax0_twin.fill_between(self.t_test_solver2, self.errors_solver2[:, state_1], color='r', alpha=0.3)
        ax0_twin.set_ylim(0, max(self.errors_solver1[:, state_1])*8)
        ticks = ax0_twin.get_yticks()
        ax0_twin.set_yticks([tick for tick in ticks if tick <= max(self.errors_solver1[:, state_1])*6])
        
        # 在左子图右上角添加时间步长标注
        ax0.text(0.95, 0.98, r"$\Delta t = " + str(time_step_ms) + " \, \mathrm{ms}$", 
                 transform=ax0.transAxes, fontsize=16, 
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', pad=0.3))

        ax1_twin.plot(self.t_test_solver1, self.errors_solver1[:, state_2], color='b', linestyle='-', alpha=0.6)
        ax1_twin.fill_between(self.t_test_solver1, self.errors_solver1[:, state_2], color='b', alpha=0.3)
        ax1_twin.plot(self.t_test_solver2, self.errors_solver2[:, state_2], color='r', linestyle='--', alpha=0.6)
        ax1_twin.fill_between(self.t_test_solver2, self.errors_solver2[:, state_2], color='r', alpha=0.3)
        ax1_twin.set_ylim(0, max(self.errors_solver1[:, state_2])*15)
        ticks = ax1_twin.get_yticks()
        ax1_twin.set_yticks([tick for tick in ticks if tick <= max(self.errors_solver1[:, state_2])*10])
        
        # 在右子图右上角添加时间步长标注
        ax1.text(0.95, 0.98, r"$\Delta t = " + str(time_step_ms) + " \, \mathrm{ms}$", 
                 transform=ax1.transAxes, fontsize=16, 
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='square', facecolor='white', edgecolor='black', pad=0.3))

        # 添加统一的图例在顶部居中
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='k', linestyle='--', label='Ground truth'),
            Line2D([0], [0], color='b', linestyle='-', linewidth=3, label='Pure solver'),
            Line2D([0], [0], color='r', linestyle='--', linewidth=2.5, label='RIA-PINN hybrid solver')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=18, frameon=True, edgecolor='black', fancybox=False)

    def add_zeros_initial_value(self, errors_simulator):
        zero_initial_errors = np.zeros((1, errors_simulator.shape[1]))
        errors_simulator = np.vstack([zero_initial_errors, errors_simulator])
        return errors_simulator
    
    def show_results(self, save_fig=False, filename='Figure_4'):
        plt.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        # 保存 PNG 版本
        save_path_png = os.path.join('outputs', f'{filename}.png')
        plt.savefig(save_path_png, dpi=150, bbox_inches='tight')
        print(f'图片已保存：{save_path_png}')
        # 保存 PDF 版本
        save_path_pdf = os.path.join('outputs', f'{filename}.pdf')
        plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
        print(f'图片已保存：{save_path_pdf}')
        plt.close()

class custom_overview2:
    def __init__(self, timestep_list):
        self.timestep_list = timestep_list
    
    def show_results(self, maximums_pure, maximums_hybrid, save_fig=False):
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(1, 2)
        ax0 = plt.subplot(gs[0, 0])
        ax0.grid()
        ax1 = plt.subplot(gs[0, 1])
        ax1.grid()
        
        # 使用与Figure_5相同的线条样式
        ax0.plot(self.timestep_list, maximums_pure[:, 0], color='b', linestyle='-', linewidth=3, alpha=0.9, label='Pure solver')
        ax0.plot(self.timestep_list, maximums_hybrid[:, 0], color='r', linestyle='--', linewidth=2.5, alpha=0.9, label='RIA-PINN hybrid solver')
        ax0.set_ylabel(r'$\ell_1$ errors', fontsize=20)
        ax0.set_xlabel(r'$\mathbf{Time\ [s]}$', fontsize=18)
        ax0.tick_params(axis='both', which='major', labelsize=16)
        
        ax1.plot(self.timestep_list, maximums_pure[:, 1], color='b', linestyle='-', linewidth=3, alpha=0.9, label='Pure solver')
        ax1.plot(self.timestep_list, maximums_hybrid[:, 1], color='r', linestyle='--', linewidth=2.5, alpha=0.9, label='RIA-PINN hybrid solver')
        ax1.set_ylabel(r'$\ell_1$ errors', fontsize=20)
        ax1.set_xlabel(r'$\mathbf{Time\ [s]}$', fontsize=18)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        
        # 统一图例在顶部居中，与Figure_5样式一致
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='b', linestyle='-', linewidth=3, label='Pure solver'),
            Line2D([0], [0], color='r', linestyle='--', linewidth=2.5, label='RIA-PINN hybrid solver')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=18, frameon=True, edgecolor='black', fancybox=False)
        
        plt.tight_layout()
        os.makedirs('outputs', exist_ok=True)
        save_path = os.path.join('outputs', 'Figure_6.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'图片已保存：{save_path}')
        plt.close()