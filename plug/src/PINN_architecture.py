import torch
import torch.nn as nn

class FCN(nn.Module):
    """Simple fully-connected network for loading legacy PINN models"""
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, range_states, lb_states):
        super().__init__()
        torch.set_default_dtype(torch.float64)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 使用 Identity 占位来匹配键名 fcs.1.weight, fcs.1.bias
        self.fcs = nn.Sequential(
            nn.Identity(),  # fcs.0 - 占位
            nn.Linear(N_INPUT, N_HIDDEN),  # fcs.1
            nn.Tanh()  # fcs.2
        )
        # fch 层：fch.0.0, fch.1.0 等
        self.fch = nn.Sequential(*[
            nn.Sequential(nn.Linear(N_HIDDEN, N_HIDDEN), nn.Tanh())
            for _ in range(N_LAYERS-1)
        ])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.range_states = torch.tensor(range_states, dtype=torch.float64).to(device)
        self.lb_states    = torch.tensor(lb_states, dtype=torch.float64).to(device)
        self.two_times    = torch.tensor([2]).to(device)
        self.one_time     = torch.tensor([1]).to(device)
    
    def forward(self, x):
        x = self.two_times * (x - self.lb_states) / self.range_states - self.one_time
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, input_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
 
    def forward(self, x):
        identity = x
        out = torch.tanh(self.fc1(x))
        out = self.fc2(out)
        out += identity
        out = torch.tanh(out)
        return out

class FCN_RESNET(nn.Module):
    "Defines a fully-connected network in PyTorch"
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, range_states, lb_states):
        super().__init__()
        torch.set_default_dtype(torch.float64)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            ResidualBlock(N_HIDDEN)]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        nn.init.xavier_normal_(self.fcs[0].weight)
        for module in self.fch:
            if isinstance(module[0], nn.Linear):
                nn.init.xavier_normal_(module[0].weight)
        nn.init.xavier_normal_(self.fce.weight)
        self.range_states = torch.tensor(range_states, dtype=torch.float64).to(device)
        self.lb_states    = torch.tensor(lb_states, dtype=torch.float64).to(device)
        self.two_times    = torch.tensor([2]).to(device)
        self.one_time     = torch.tensor([1]).to(device)
    def forward(self, x):
        x = self.two_times * (x - self.lb_states) / self.range_states - self.one_time
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x