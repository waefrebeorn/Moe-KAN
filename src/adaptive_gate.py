import torch
from torch import nn
from efficient_kan.kan import KAN

class AdaptiveGate(nn.Module):
    """
    Adaptive gating mechanism for Mixture of Experts.

    Args:
        input_dim (int): Dimensionality of the input.
        num_experts (int): Number of expert networks.
    """
    def __init__(self, input_dim, num_experts):
        super(AdaptiveGate, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        """
        Forward pass through the adaptive gate.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Gate weights for each expert of shape (batch_size, num_experts).
        """
        gate_weights = torch.softmax(self.fc(x), dim=1)
        return gate_weights

class MoEWithAdaptiveGate(nn.Module):
    """
    Mixture of Experts model with an adaptive gating mechanism.

    Args:
        dim (int): Dimensionality of the input.
        num_experts (int): Number of expert networks.
    """
    def __init__(self, dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.gate = AdaptiveGate(dim, num_experts)
        self.experts = nn.ModuleList([KAN(dim, num_layers=3, num_hidden=128) for _ in range(self.num_experts)])
    
    def forward(self, x):
        """
        Forward pass through the MoE-KAN model with adaptive gate.

        Args:
            x (Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            Tensor: Output tensor after passing through experts and gating network.
        """
        gate_weights = self.gate(x)
        expert_outputs = torch.stack([self.experts[i](x) for i in range(self.num_experts)], dim=1)
        output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)
        return output
