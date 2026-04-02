"""
Data handler for IEEE 9-bus DAE-PINN training
"""
import torch
import numpy as np
import deepxde as dde


class IEEE9BusDataHandler:
    """
    Handles training and test data generation
    
    Similar to DAE-PINNs, uses random sampling in state space
    """
    
    def __init__(
        self,
        num_train=10000,
        num_test=1000,
        num_IRK_stages=10,
        state_dim=12,
        state_space_range=None,
        device='cpu',
    ):
        self.num_train = num_train
        self.num_test = num_test
        self.num_IRK_stages = num_IRK_stages
        self.state_dim = state_dim
        self.device = device
        
        # Default state space range (normalized)
        if state_space_range is None:
            # [E'q, E'd, δ, ω] for each generator
            # Typical ranges:
            # E'q, E'd: [0.8, 1.2] p.u.
            # δ: [-0.5, 0.5] rad
            # ω: [-0.01, 0.01] rad/s (deviation from nominal)
            state_space_range = [(-0.5, 0.5)] * state_dim
        
        self.state_space_range = state_space_range
        
        # Load IRK weights
        self._load_IRK_weights()
        
        # Generate training and test data
        self._generate_data()
    
    def _load_IRK_weights(self):
        """Load IRK weights from file or generate default"""
        IRK_file = f'./data/IRK_weights/Butcher_IRK{self.num_IRK_stages}.txt'
        
        try:
            tmp = np.loadtxt(IRK_file, ndmin=2)
            IRK_weights = np.reshape(
                tmp[0:self.num_IRK_stages**2 + self.num_IRK_stages],
                (self.num_IRK_stages + 1, self.num_IRK_stages)
            )
            self.IRK_weights = torch.tensor(IRK_weights, dtype=torch.float32).to(self.device)
            self.IRK_times = tmp[self.num_IRK_stages**2 + self.num_IRK_stages:]
        except FileNotFoundError:
            print(f"Warning: IRK weights file not found at {IRK_file}")
            print("Using default Radau IIA weights...")
            
            # Use default Radau IIA weights (simplified)
            self.IRK_weights = torch.ones(
                self.num_IRK_stages + 1, self.num_IRK_stages
            ).to(self.device) / self.num_IRK_stages
            
            self.IRK_times = np.linspace(0, 1, self.num_IRK_stages)
    
    def _generate_data(self):
        """Generate random training and test points"""
        # Create geometry for sampling
        mins = [r[0] for r in self.state_space_range]
        maxs = [r[1] for r in self.state_space_range]
        
        geom = dde.geometry.Hypercube(mins, maxs)
        
        # Generate training points
        np.random.seed(1234)
        self.X_train = geom.random_points(self.num_train)
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        
        # Generate test points
        np.random.seed(3456)
        self.X_test = geom.random_points(self.num_test)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
        
        print(f"Generated {self.num_train} training points and {self.num_test} test points")
    
    def get_train_batch(self, batch_size=None):
        """Get a batch of training data"""
        if batch_size is None or batch_size >= self.num_train:
            return self.X_train
        
        indices = np.random.choice(self.num_train, batch_size, replace=False)
        return self.X_train[indices]
    
    def get_test_data(self):
        """Get all test data"""
        return self.X_test
    
    def get_IRK_weights(self):
        """Get IRK weights"""
        return self.IRK_weights
