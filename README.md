# Moe-KAN

Moe-KAN is a project that integrates Kolmogorov–Arnold Networks (KANs) into a Mixture of Experts (MoE) framework, inspired by the simplicity and efficiency of combining these two approaches.

## Inspiration
- **Frazer** on Discord suggested the idea by stating "you can just combine them."
- This project takes inspiration from:
  - [st-moe-pytorch](https://github.com/lucidrains/st-moe-pytorch)
  - [efficient-kan](https://github.com/Blealtan/efficient-kan)

## Structure
- **src/moe_kan.py**: Contains the MoE implementation with KAN experts.
- **src/efficient_kan/kan.py**: Contains the efficient KAN implementation.

## Usage
```python
import torch
from src.moe_kan import MoE

# Model parameters
input_dim = 128
output_dim = 10
num_experts = 4
top_k = 2

# Instantiate the MoE model
moe_model = MoE(input_dim, num_experts)

# Example input
input_data = torch.randn(32, input_dim)  # batch size of 32

# Forward pass
output_data = moe_model(input_data)

print(output_data.shape)  # Should be [32, input_dim]
```

## Requirements
- torch
- efficient-kan (imported directly in the project)

Install the requirements using:
```bash
pip install torch
```

## Contributing
Feel free to submit issues or pull requests.

## Credits
- [st-moe-pytorch](https://github.com/lucidrains/st-moe-pytorch)
- [efficient-kan](https://github.com/Blealtan/efficient-kan)
- Idea inspired by Frazer on Discord
```

