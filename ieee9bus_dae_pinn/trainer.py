"""
Training script for IEEE 9-bus DAE-PINN
"""
import os
import time
import torch
import numpy as np
from tqdm import tqdm

from models import IEEE9Bus_PINN
from physics import IEEE9BusPhysics, dotdict, compute_total_loss
from data_handler import IEEE9BusDataHandler


class Trainer:
    """
    Trainer class for IEEE 9-bus PINN
    
    Similar to DAE-PINNs supervisor approach
    """
    
    def __init__(
        self,
        config_dynamic_path='../config_files/config_machines_dynamic.yaml',
        config_static_path='../config_files/config_machines_static.yaml',
        Y_admittance_path='../config_files/network_admittance.pt',
        log_dir='./logs',
        device='cuda',
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Load physics constraints
        self.physics = IEEE9BusPhysics(
            config_dynamic_path,
            config_static_path,
            Y_admittance_path
        )
        
        # Initialize data handler
        self.data_handler = None
        
        # Initialize model
        self.model = None
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.loss_history = []
        self.best_loss = float('inf')
    
    def setup_model(
        self,
        num_IRK_stages=10,
        dyn_layer_size=None,
        alg_layer_size=None,
        activation='tanh',
        stacked=True,
    ):
        """Setup neural network model"""
        
        # Default layer sizes
        if dyn_layer_size is None:
            dyn_layer_size = [12, 64, 64, 64, num_IRK_stages + 1]
        
        if alg_layer_size is None:
            alg_layer_size = [12, 64, 64, 18 * (num_IRK_stages + 1)]
        
        # Dynamic network config
        dynamic = dotdict({
            'num_IRK_stages': num_IRK_stages,
            'layer_size': dyn_layer_size,
            'activation': activation,
            'initializer': 'Glorot normal',
            'dropout_rate': None,
        })
        
        # Algebraic network config
        algebraic = dotdict({
            'num_IRK_stages': num_IRK_stages,
            'layer_size': alg_layer_size,
            'activation': activation,
            'initializer': 'Glorot normal',
            'dropout_rate': None,
        })
        
        # Optional: add input feature transformation
        def dyn_input_feature_layer(x):
            """Fourier feature encoding"""
            return torch.cat([
                x,
                torch.sin(np.pi * x),
                torch.cos(np.pi * x),
                torch.sin(2 * np.pi * x),
                torch.cos(2 * np.pi * x),
            ], dim=-1)
        
        def alg_output_feature_layer(x):
            """Ensure positive voltage magnitudes"""
            return torch.nn.functional.softplus(x)
        
        # Create model
        self.model = IEEE9Bus_PINN(
            dynamic,
            algebraic,
            stacked=stacked,
            dyn_in_transform=None,  # Can enable dyn_input_feature_layer
            alg_out_transform=alg_output_feature_layer,
        ).to(self.device)
        
        print(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def setup_data(
        self,
        num_train=10000,
        num_test=1000,
        num_IRK_stages=10,
    ):
        """Setup data handler"""
        self.data_handler = IEEE9BusDataHandler(
            num_train=num_train,
            num_test=num_test,
            num_IRK_stages=num_IRK_stages,
            state_dim=12,
            device=self.device,
        )
    
    def setup_optimizer(
        self,
        lr=1e-3,
        scheduler_type=None,
        patience=1000,
        factor=0.5,
    ):
        """Setup optimizer and learning rate scheduler"""
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        if scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=patience,
                factor=factor,
                verbose=True,
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=patience,
                gamma=factor,
            )
        else:
            self.scheduler = None
    
    def train(
        self,
        epochs=10000,
        batch_size=None,
        h=0.04,  # time step size
        loss_weights=[1.0, 1.0],  # [dynamic weight, algebraic weight]
        test_every=100,
        save_every=1000,
        model_name='ieee9bus_pinn',
    ):
        """
        Main training loop
        
        Args:
            epochs: number of training epochs
            batch_size: batch size (None = full batch)
            h: time step size
            loss_weights: weights for dynamic and algebraic losses
            test_every: test interval
            save_every: model save interval
            model_name: model name for saving
        """
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Loss weights: dynamic={loss_weights[0]}, algebraic={loss_weights[1]}")
        
        # Get IRK weights
        IRK_weights = self.data_handler.get_IRK_weights()
        h_tensor = torch.tensor([h], dtype=torch.float32).to(self.device)
        
        start_time = time.time()
        
        for epoch in tqdm(range(epochs)):
            # Get training batch
            X_batch = self.data_handler.get_train_batch(batch_size)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Compute physics residuals
            f_residuals, g_residuals = self.physics.compute_IRK_residuals(
                self.model, X_batch, h_tensor, IRK_weights, self.device
            )
            
            # Compute loss
            total_loss, loss_dict = compute_total_loss(
                f_residuals, g_residuals, loss_weights
            )
            
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(total_loss)
                else:
                    self.scheduler.step()
            
            # Record loss
            self.loss_history.append(loss_dict)
            
            # Test and save
            if (epoch + 1) % test_every == 0:
                test_loss = self._test(epoch, h_tensor, IRK_weights, loss_weights)
                
                if test_loss < self.best_loss:
                    self.best_loss = test_loss
                    self._save_model(model_name, epoch, is_best=True)
            
            if (epoch + 1) % save_every == 0:
                self._save_model(model_name, epoch, is_best=False)
        
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time:.2f} seconds")
        print(f"Best loss: {self.best_loss:.6e}")
    
    def _test(self, epoch, h, IRK_weights, loss_weights):
        """Test on validation set"""
        self.model.eval()
        
        with torch.no_grad():
            X_test = self.data_handler.get_test_data()
            
            f_residuals, g_residuals = self.physics.compute_IRK_residuals(
                self.model, X_test, h, IRK_weights, self.device
            )
            
            total_loss, loss_dict = compute_total_loss(
                f_residuals, g_residuals, loss_weights
            )
        
        print(f"\nEpoch {epoch}: Test loss = {total_loss.item():.6e}, "
              f"Dyn = {loss_dict['loss_dyn']:.6e}, Alg = {loss_dict['loss_alg']:.6e}")
        
        self.model.train()
        return total_loss.item()
    
    def _save_model(self, model_name, epoch, is_best):
        """Save model checkpoint"""
        if is_best:
            filename = os.path.join(self.log_dir, f'{model_name}_best.pth')
        else:
            filename = os.path.join(self.log_dir, f'{model_name}_epoch{epoch}.pth')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'best_loss': self.best_loss,
        }, filename)
        
        if is_best:
            print(f"  → Saved best model (loss: {self.best_loss:.6e})")
    
    def load_model(self, model_path):
        """Load model checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_history = checkpoint['loss_history']
        self.best_loss = checkpoint['best_loss']
        
        print(f"Loaded model from {model_path}")
        print(f"  Epoch: {checkpoint['epoch']}, Best loss: {self.best_loss:.6e}")
