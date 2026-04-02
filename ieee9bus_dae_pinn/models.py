"""
Neural Network Models for IEEE 9-Bus DAE-PINN
Based on DAE-PINNs architecture
"""
import torch
import torch.nn as nn


class FNN(nn.Module):
    """Feedforward Neural Network"""
    
    def __init__(
        self, 
        layer_size, 
        activation='tanh',
        initializer='Glorot normal',
        dropout_rate=None,
        input_transform=None,
        output_transform=None,
    ):
        super(FNN, self).__init__()
        
        self.input_transform = input_transform
        self.output_transform = output_transform
        
        # Build network layers
        layers = []
        for i in range(len(layer_size) - 1):
            layers.append(nn.Linear(layer_size[i], layer_size[i+1]))
            
            # Add activation (not for output layer)
            if i < len(layer_size) - 2:
                if activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'sin':
                    layers.append(torch.sin)
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                
                # Add dropout
                if dropout_rate:
                    layers.append(nn.Dropout(dropout_rate))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights(initializer)
    
    def _initialize_weights(self, initializer):
        if initializer == 'Glorot normal':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        elif initializer == 'Glorot uniform':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        
        output = self.net(x)
        
        if self.output_transform is not None:
            output = self.output_transform(output)
        
        return output


class IEEE9Bus_PINN(nn.Module):
    """
    PINN for IEEE 9-bus 3-machine power system
    
    Based on DAE-PINNs architecture, adapted for IEEE 9-bus system:
    - 12 dynamic states (E'q, E'd, δ, ω for each of 3 generators)
    - 18 algebraic variables (V, θ for each of 9 buses)
    
    Architecture:
    - Dynamic networks: predict IRK stages for each state variable
    - Algebraic network: predict IRK stages for algebraic variables
    """
    
    def __init__(
        self,
        dynamic_config,
        algebraic_config,
        stacked=True,
        dyn_in_transform=None,
        dyn_out_transform=None,
        alg_in_transform=None,
        alg_out_transform=None,
    ):
        super(IEEE9Bus_PINN, self).__init__()
        
        self.stacked = stacked
        self.num_generators = 3
        self.states_per_gen = 4  # E'q, E'd, δ, ω
        self.dim_dynamic = self.num_generators * self.states_per_gen  # 12
        self.num_IRK_stages = dynamic_config.num_IRK_stages
        
        # Build dynamic state networks
        if stacked:
            # Create 12 separate networks (one for each state variable)
            self.Y = nn.ModuleList([
                FNN(
                    dynamic_config.layer_size,
                    dynamic_config.activation,
                    dynamic_config.initializer,
                    dropout_rate=dynamic_config.dropout_rate,
                    input_transform=dyn_in_transform,
                    output_transform=dyn_out_transform,
                )
                for _ in range(self.dim_dynamic)
            ])
        else:
            # Create 1 combined network
            dim_out = self.dim_dynamic * (self.num_IRK_stages + 1)
            layer_size = dynamic_config.layer_size.copy()
            layer_size[-1] = dim_out
            self.Y = FNN(
                layer_size,
                dynamic_config.activation,
                dynamic_config.initializer,
                dropout_rate=dynamic_config.dropout_rate,
                input_transform=dyn_in_transform,
                output_transform=dyn_out_transform,
            )
        
        # Build algebraic variable network
        self.num_buses = 9
        self.dim_algebraic = 2 * self.num_buses  # V and θ for each bus
        alg_layer_size = algebraic_config.layer_size.copy()
        alg_layer_size[-1] = self.dim_algebraic * (self.num_IRK_stages + 1)
        
        self.Z = FNN(
            alg_layer_size,
            algebraic_config.activation,
            algebraic_config.initializer,
            dropout_rate=algebraic_config.dropout_rate,
            input_transform=alg_in_transform,
            output_transform=alg_out_transform,
        )
    
    def forward(self, input):
        """
        Forward pass for IEEE 9-bus system
        
        Args:
            input: Tensor of shape [batch_size, input_dim]
        
        Returns:
            Y_outputs: list of IRK stages for dynamic states
            Z_outputs: list of IRK stages for algebraic variables
        """
        dim_out = self.num_IRK_stages + 1
        
        # Process dynamic states
        Y_outputs = []
        if self.stacked:
            # Each state has its own network
            for i in range(self.dim_dynamic):
                Yi = self.Y[i](input)  # [batch_size, num_IRK_stages + 1]
                Y_outputs.append(Yi)
        else:
            # Single network outputs all states
            Y_all = self.Y(input)  # [batch_size, dim * (num_IRK_stages + 1)]
            for i in range(self.dim_dynamic):
                Yi = Y_all[..., i * dim_out:(i + 1) * dim_out]
                Y_outputs.append(Yi)
        
        # Process algebraic variables
        Z = self.Z(input)  # [batch_size, algebraic_dim * (num_IRK_stages + 1)]
        
        # Reshape Z to separate variables
        Z_outputs = []
        for i in range(self.dim_algebraic):
            Zi = Z[..., i * dim_out:(i + 1) * dim_out]
            Z_outputs.append(Zi)
        
        return Y_outputs, Z_outputs
