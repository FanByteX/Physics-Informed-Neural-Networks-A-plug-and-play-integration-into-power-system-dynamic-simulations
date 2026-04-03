"""
Physics constraints for IEEE 9-bus system (3-machine)
Based on DAE-PINNs approach + plug tds_dae_rk_schemes.py

State vector layout (input to PINN, dim=12):
  [0]  E'q_1,  [1]  E'd_1,  [2]  d_1,  [3]  w_1
  [4]  E'q_2,  [5]  E'd_2,  [6]  d_2,  [7]  w_2
  [8]  E'q_3,  [9]  E'd_3,  [10] d_3,  [11] w_3

Algebraic vector layout (Z network output, dim=18):
  [0] V_1, [1] th_1, [2] V_2, [3] th_2, ... [16] V_9, [17] th_9
  (first 6 entries = generator buses 1-3)

Dynamic equations (per generator, classical model):
  dE'q/dt = 0
  dE'd/dt = 0
  dd/dt   = w * 2*pi*f
  dw/dt   = (Pg - Pe - D*w) / (2*H)
    Pe = E'q*Iq + E'd*Id
    Id = (E'q - V*cos(d-th)) / X'd
    Iq = -(E'd - V*sin(d-th)) / X'd

Algebraic equations (per generator, at next time step):
  KCL: stator current in network frame == Y-matrix injection
    I_net = (Id + j*Iq) * exp(j*(d - pi/2))
    I_inj = sum_k Y[i,k] * V_k * exp(j*th_k)
  => Re(I_net) - Re(I_inj) = 0
     Im(I_net) - Im(I_inj) = 0
"""
import torch
import numpy as np
import yaml


class dotdict(dict):
    """dot-notation dict"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class IEEE9BusPhysics:
    def __init__(self, config_dynamic_path, config_static_path, Y_admittance_path):
        self._load_parameters(config_dynamic_path, config_static_path, Y_admittance_path)
        self.num_generators = 3
        self.num_buses      = 9
        self.states_per_gen = 4
        self.dim_dynamic    = 12
        self.dim_algebraic  = 18

    # ------------------------------------------------------------------
    def _load_parameters(self, dynamic_path, static_path, admittance_path):
        with open(dynamic_path, 'r') as f:
            dyn = yaml.safe_load(f)
        self.freq = torch.tensor(float(dyn['freq']),                        dtype=torch.float32)
        self.H    = torch.tensor(list(dyn['inertia_H'].values()),           dtype=torch.float32)
        self.Rs   = torch.tensor(list(dyn['Rs'].values()),                  dtype=torch.float32)
        self.Xd_p = torch.tensor(list(dyn['Xd_prime'].values()),            dtype=torch.float32)
        self.Pg   = torch.tensor(list(dyn['Pg_setpoints'].values()),        dtype=torch.float32)
        self.D    = torch.tensor(list(dyn['Damping_D'].values()),           dtype=torch.float32)

        with open(static_path, 'r') as f:
            sta = yaml.safe_load(f)
        self.V_mag   = torch.tensor(list(sta['Voltage_magnitude'].values()), dtype=torch.float32)
        self.V_angle = torch.tensor(list(sta['Voltage_angle'].values()),     dtype=torch.float32)
        self.Xd      = torch.tensor(list(sta['Xd'].values()),                dtype=torch.float32)
        self.Xq      = torch.tensor(list(sta['Xq'].values()),                dtype=torch.float32)
        self.Xq_p    = torch.tensor(list(sta['Xq_prime'].values()),          dtype=torch.float32)

        self.Y_adm = torch.load(admittance_path, map_location='cpu')

    # ------------------------------------------------------------------
    # Stator equations
    # ------------------------------------------------------------------
    def _stator_Id(self, Eq_p, V, delta, theta, gen, device):
        xdp = self.Xd_p[gen].to(device)
        return (Eq_p - V * torch.cos(delta - theta)) / xdp

    def _stator_Iq(self, Ed_p, V, delta, theta, gen, device):
        xdp = self.Xd_p[gen].to(device)
        return -(Ed_p - V * torch.sin(delta - theta)) / xdp

    # ------------------------------------------------------------------
    # Reference-frame transform  I_net = (Id+j*Iq)*exp(j*(delta-pi/2))
    # ------------------------------------------------------------------
    def _ref_transform(self, Id, Iq, delta, device):
        angle = delta - np.pi / 2
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        return Id * cos_a - Iq * sin_a, Id * sin_a + Iq * cos_a

    # ------------------------------------------------------------------
    # Network KCL injection  I_inj_i = sum_k Y[i,k]*V_k*exp(j*th_k)
    # ------------------------------------------------------------------
    def _network_injection(self, V_list, Th_list, gen, device):
        Y = self.Y_adm.to(device)
        I_re = torch.zeros_like(V_list[0])
        I_im = torch.zeros_like(V_list[0])
        for k in range(self.num_generators):
            Vk_re = V_list[k] * torch.cos(Th_list[k])
            Vk_im = V_list[k] * torch.sin(Th_list[k])
            Yre = Y[gen, k].real.to(device)
            Yim = Y[gen, k].imag.to(device)
            I_re = I_re + Yre * Vk_re - Yim * Vk_im
            I_im = I_im + Yim * Vk_re + Yre * Vk_im
        return I_re, I_im

    # ------------------------------------------------------------------
    # IRK residual:  x_{n+1} - x_n - h * F_stages @ b^T
    # x_stages [batch, s+1]: cols 0..s-1 = stage values, col s = next step
    # F_stages [batch, s]  : rhs at each stage
    # IRK_weights [s+1, s] : last row = b (quadrature weights)
    # ------------------------------------------------------------------
    def _irk_res(self, x_stages, x_0, F_stages, h, IRK_weights, device):
        b = IRK_weights[-1:, :].to(device)              # [1, s]
        weighted = F_stages.mm(b.T)                     # [batch, 1]
        return x_stages[..., -1:] - x_0 - h * weighted

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def compute_IRK_residuals(self, model, inputs, h, IRK_weights, device='cpu'):
        """
        inputs      : [batch, 12]
        h           : scalar tensor
        IRK_weights : [s+1, s]
        Returns f_residuals (list of 12 [batch,1]) and
                g_residuals (list of  6 [batch,1])
        """
        Y_out, Z_out = model(inputs)
        # Y_out: 12 tensors of [batch, s+1]
        # Z_out: 18 tensors of [batch, s+1]
        s = IRK_weights.shape[1]

        f_res = []
        g_res = []

        # IRK stage columns (0..s-1) for generator voltages & angles
        V_s  = [Z_out[2 * g][..., :s]     for g in range(self.num_generators)]
        Th_s = [Z_out[2 * g + 1][..., :s] for g in range(self.num_generators)]

        for gen in range(self.num_generators):
            bi = gen * self.states_per_gen

            # --- initial state ---
            Eq0 = inputs[..., bi + 0:bi + 1]
            Ed0 = inputs[..., bi + 1:bi + 2]
            d0  = inputs[..., bi + 2:bi + 3]
            w0  = inputs[..., bi + 3:bi + 4]

            # --- IRK stage values ---
            Eq_s = Y_out[bi + 0][..., :s]
            Ed_s = Y_out[bi + 1][..., :s]
            d_s  = Y_out[bi + 2][..., :s]
            om_s = Y_out[bi + 3][..., :s]

            Vs  = V_s[gen]
            Ths = Th_s[gen]

            # ---- f1: dE'q/dt = 0 ----
            f_res.append(Y_out[bi + 0][..., -1:] - Eq0)

            # ---- f2: dE'd/dt = 0 ----
            f_res.append(Y_out[bi + 1][..., -1:] - Ed0)

            # ---- f3: dd/dt = w * 2*pi*f ----
            F_d = om_s * 2.0 * np.pi * self.freq.to(device)
            f_res.append(self._irk_res(Y_out[bi + 2], d0, F_d, h, IRK_weights, device))

            # ---- f4: dw/dt = (Pg - Pe - D*w) / (2H) ----
            Id_s  = self._stator_Id(Eq_s, Vs, d_s, Ths, gen, device)
            Iq_s  = self._stator_Iq(Ed_s, Vs, d_s, Ths, gen, device)
            Pe_s  = Eq_s * Iq_s + Ed_s * Id_s
            F_w   = (self.Pg[gen].to(device) - Pe_s - self.D[gen].to(device) * om_s) / (2.0 * self.H[gen].to(device))
            f_res.append(self._irk_res(Y_out[bi + 3], w0, F_w, h, IRK_weights, device))

            # ---- algebraic residuals at NEXT time step ----
            Eq_n  = Y_out[bi + 0][..., -1:]
            Ed_n  = Y_out[bi + 1][..., -1:]
            d_n   = Y_out[bi + 2][..., -1:]
            V_n   = Z_out[2 * gen][..., -1:]
            Th_n  = Z_out[2 * gen + 1][..., -1:]

            Id_n = self._stator_Id(Eq_n, V_n, d_n, Th_n, gen, device)
            Iq_n = self._stator_Iq(Ed_n, V_n, d_n, Th_n, gen, device)

            Inet_re, Inet_im = self._ref_transform(Id_n, Iq_n, d_n, device)

            V_n_all  = [Z_out[2 * g][..., -1:]     for g in range(self.num_generators)]
            Th_n_all = [Z_out[2 * g + 1][..., -1:] for g in range(self.num_generators)]
            Iinj_re, Iinj_im = self._network_injection(V_n_all, Th_n_all, gen, device)

            g_res.append(Inet_re - Iinj_re)
            g_res.append(Inet_im - Iinj_im)

        return f_res, g_res


# --------------------------------------------------------------------------
def mse_loss(r):
    return torch.mean(r ** 2)


def compute_total_loss(f_residuals, g_residuals, weights=None):
    if weights is None:
        weights = [1.0, 1.0]
    loss_dyn = sum(mse_loss(f) for f in f_residuals)
    loss_alg = sum(mse_loss(g) for g in g_residuals)
    total    = weights[0] * loss_dyn + weights[1] * loss_alg
    return total, {
        'loss_dyn':   loss_dyn.item(),
        'loss_alg':   loss_alg.item(),
        'loss_total': total.item(),
    }
