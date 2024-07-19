import torch
from torch import nn
from efficient_kan.kan import KAN

class MoE(nn.Module):
    """
    Mixture of Experts model that integrates Kolmogorov–Arnold Networks (KANs).

    Args:
        dim (int): Dimensionality of the input.
        num_experts (int): Number of expert networks.
    """
    def __init__(self, dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([KAN(dim, num_layers=3, num_hidden=128) for _ in range(self.num_experts)])
    
    def forward(self, x):
        """
        Forward pass through the MoE-KAN model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            Tensor: Output tensor after passing through experts and gating network.
        """
        gate_outputs = self.gate(x)
        topk_vals, topk_indices = torch.topk(gate_outputs, 2, dim=1)
        topk_vals = torch.softmax(topk_vals, dim=1).unsqueeze(-1)
        
        expert_outputs = torch.stack([self.experts[i](x) for i in range(self.num_experts)], dim=1)
        topk_expert_outputs = torch.gather(expert_outputs, 1, topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1)))
        
        output = torch.sum(topk_expert_outputs * topk_vals, dim=1)
        return output
