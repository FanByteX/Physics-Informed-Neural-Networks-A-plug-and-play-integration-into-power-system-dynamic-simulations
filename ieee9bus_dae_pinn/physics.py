"""
Physics constraints for IEEE 9-bus system
Based on DAE-PINNs approach
"""
import torch
import numpy as np
import yaml


class dotdict(dict):
    """Dictionary that supports dot notation access"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class IEEE9BusPhysics:
    """
    Physics constraints for IEEE 9-bus 3-machine system
    
    Implements:
    - Dynamic equations (swing equations, flux decay)
    - Algebraic equations (network power flow)
    - IRK residual computation
    """
    
    def __init__(self, config_dynamic_path, config_static_path, Y_admittance_path):
        """Load system parameters"""
        self._load_parameters(config_dynamic_path, config_static_path, Y_admittance_path)
        
        # System dimensions
        self.num_generators = 3
        self.num_buses = 9
        self.states_per_gen = 4  # E'q, E'd, δ, ω
        self.dim_dynamic = self.num_generators * self.states_per_gen  # 12
        self.dim_algebraic = 2 * self.num_buses  # 18
    
    def _load_parameters(self, dynamic_path, static_path, admittance_path):
        """Load system parameters from config files"""
        # Load dynamic parameters
        with open(dynamic_path, 'r') as f:
            dynamic_config = yaml.safe_load(f)
        
        self.freq = torch.tensor(dynamic_config['freq'], dtype=torch.float32)
        self.H = torch.tensor(list(dynamic_config['inertia_H'].values()), dtype=torch.float32)
        self.Rs = torch.tensor(list(dynamic_config['Rs'].values()), dtype=torch.float32)
        self.Xd_prime = torch.tensor(list(dynamic_config['Xd_prime'].values()), dtype=torch.float32)
        self.Pg_setpoints = torch.tensor(list(dynamic_config['Pg_setpoints'].values()), dtype=torch.float32)
        self.Damping = torch.tensor(list(dynamic_config['Damping_D'].values()), dtype=torch.float32)
        
        # Load static parameters
        with open(static_path, 'r') as f:
            static_config = yaml.safe_load(f)
        
        self.V_mag = torch.tensor(list(static_config['Voltage_magnitude'].values()), dtype=torch.float32)
        self.V_angle = torch.tensor(list(static_config['Voltage_angle'].values()), dtype=torch.float32)
        self.Xd = torch.tensor(list(static_config['Xd'].values()), dtype=torch.float32)
        self.Xq = torch.tensor(list(static_config['Xq'].values()), dtype=torch.float32)
        self.Xq_prime = torch.tensor(list(static_config['Xq_prime'].values()), dtype=torch.float32)
        
        # Load admittance matrix
        self.Y_admittance = torch.load(admittance_path, map_location='cpu')
    
    def compute_IRK_residuals(
        self, 
        model, 
        inputs, 
        h, 
        IRK_weights,
        device='cpu'
    ):
        """
        Compute IRK residuals for dynamic and algebraic equations
        
        Args:
            model: neural network model
            inputs: state variables [batch_size, dim_dynamic]
            h: time step size
            IRK_weights: IRK weights [num_stages+1, num_stages]
            device: torch device
        
        Returns:
            f_residuals: list of dynamic equation residuals
            g_residuals: list of algebraic equation residuals
        """
        # Get IRK stage predictions from model
        Y_outputs, Z_outputs = model(inputs)
        
        # Extract dynamic states for each generator
        # Y_outputs[0-3]: Generator 1 (E'q1, E'd1, δ1, ω1)
        # Y_outputs[4-7]: Generator 2 (E'q2, E'd2, δ2, ω2)
        # Y_outputs[8-11]: Generator 3 (E'q3, E'd3, δ3, ω3)
        
        f_residuals = []
        
        # Compute residuals for each generator
        for gen in range(self.num_generators):
            base_idx = gen * self.states_per_gen
            
            Eq_prime_stages = Y_outputs[base_idx + 0]  # E'q IRK stages
            Ed_prime_stages = Y_outputs[base_idx + 1]  # E'd IRK stages
            delta_stages = Y_outputs[base_idx + 2]     # δ IRK stages
            omega_stages = Y_outputs[base_idx + 3]     # ω IRK stages
            
            # Extract current stage values (last stage is current time)
            Eq_prime = Eq_prime_stages[..., :-1]
            Ed_prime = Ed_prime_stages[..., :-1]
            delta = delta_stages[..., :-1]
            omega = omega_stages[..., :-1]
            
            # Extract current state values
            Eq_prime_0 = inputs[..., base_idx + 0:base_idx + 1]
            Ed_prime_0 = inputs[..., base_idx + 1:base_idx + 2]
            delta_0 = inputs[..., base_idx + 2:base_idx + 3]
            omega_0 = inputs[..., base_idx + 3:base_idx + 4]
            
            # Compute swing equation residuals
            # dδ/dt = ω * 2πf
            f_delta = self._swing_eq_delta(
                delta_stages, omega_stages, delta_0, omega_0, h, IRK_weights, gen, device
            )
            
            # dω/dt = (Pg - Pe - D*ω) / (2H)
            f_omega = self._swing_eq_omega(
                omega_stages, Eq_prime, Ed_prime, delta, omega, omega_0, h, IRK_weights, gen, device
            )
            
            # E'q and E'd are constant for classical model (no flux decay)
            f_Eq = Eq_prime_stages[..., -1:] - Eq_prime_0
            f_Ed = Ed_prime_stages[..., -1:] - Ed_prime_0
            
            f_residuals.extend([f_Eq, f_Ed, f_delta, f_omega])
        
        # Compute algebraic equation residuals (simplified)
        # In practice, you would need to implement full power flow equations
        g_residuals = self._compute_algebraic_residuals(Z_outputs, Y_outputs, device)
        
        return f_residuals, g_residuals
    
    def _swing_eq_delta(self, delta_stages, omega_stages, delta_0, omega_0, h, IRK_weights, gen, device):
        """Compute swing equation residual for δ"""
        omega = omega_stages[..., :-1]
        delta = delta_stages[..., :-1]
        
        # T = 1.0 (time scale factor)
        T = 1.0
        F_delta = T * omega * 2 * np.pi * self.freq.to(device)
        
        # IRK residual: δ_1 - δ_0 - h * Σ(F_i * w_i)
        residual = delta_stages[..., -1:] - delta_0 - h * F_delta.mm(IRK_weights.T.to(device))
        
        return residual
    
    def _swing_eq_omega(self, omega_stages, Eq_prime, Ed_prime, delta, omega, omega_0, h, IRK_weights, gen, device):
        """Compute swing equation residual for ω"""
        # Simplified power calculation (would need full network equations in practice)
        # Pe = Eq_prime * Id + Ed_prime * Iq
        
        # For now, use simplified model
        T = 1.0
        Pg = self.Pg_setpoints[gen].to(device)
        D = self.Damping[gen].to(device)
        H = self.H[gen].to(device)
        
        # Placeholder: would need actual current calculations
        Pe = torch.zeros_like(omega).to(device)
        
        F_omega = T * (Pg - Pe - D * omega) / (2 * H)
        
        # IRK residual
        residual = omega_stages[..., -1:] - omega_0 - h * F_omega.mm(IRK_weights.T.to(device))
        
        return residual
    
    def _compute_algebraic_residuals(self, Z_outputs, Y_outputs, device):
        """
        Compute algebraic equation residuals
        
        Power flow equations for each bus:
        - Real power balance: P_gen - P_load - P_flow = 0
        - Reactive power balance: Q_gen - Q_load - Q_flow = 0
        """
        g_residuals = []
        
        # Simplified version: ensure voltage magnitudes are reasonable
        for bus in range(self.num_buses):
            V_idx = bus * 2
            theta_idx = bus * 2 + 1
            
            V = Z_outputs[V_idx][..., -1:]  # Voltage magnitude
            # theta = Z_outputs[theta_idx][..., -1:]  # Voltage angle
            
            # Simple constraint: voltage should be close to 1.0 p.u.
            g_V = (V - 1.0) ** 2
            g_residuals.append(g_V)
        
        return g_residuals


def mse_loss(residual):
    """Mean squared error loss"""
    return torch.mean(residual ** 2)


def compute_total_loss(f_residuals, g_residuals, weights=None):
    """
    Compute total physics-informed loss
    
    Args:
        f_residuals: list of dynamic equation residuals
        g_residuals: list of algebraic equation residuals
        weights: optional [weight_dyn, weight_alg]
    
    Returns:
        total_loss, loss_dict
    """
    if weights is None:
        weights = [1.0, 1.0]
    
    # Dynamic losses
    loss_dyn = sum([mse_loss(f) for f in f_residuals])
    
    # Algebraic losses
    loss_alg = sum([mse_loss(g) for g in g_residuals])
    
    # Total weighted loss
    total_loss = weights[0] * loss_dyn + weights[1] * loss_alg
    
    loss_dict = {
        'loss_dyn': loss_dyn.item(),
        'loss_alg': loss_alg.item(),
        'loss_total': total_loss.item(),
    }
    
    return total_loss, loss_dict
