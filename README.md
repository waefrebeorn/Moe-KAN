# Moe-KAN

Moe-KAN integrates Kolmogorov–Arnold Networks (KANs) into a Mixture of Experts (MoE) framework, enhancing efficiency and performance. Inspired by a suggestion from Frazer on Discord, this project combines the strengths of MoE and KANs for advanced machine learning tasks.

## Table of Contents
- [Introduction](#introduction)
- [Inspiration](#inspiration)
- [Structure](#structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Examples](#examples)
- [Diagrams](#diagrams)
- [Expand Features](#expand-features)
- [Contributing](#contributing)
- [Credits](#credits)
- [License](#license)

## Introduction
Moe-KAN leverages the power of Kolmogorov–Arnold Networks within a Mixture of Experts framework to provide a scalable and efficient solution for complex machine learning problems. This combination allows for dynamic selection of the most suitable expert for a given input, improving both accuracy and computational efficiency.

## Inspiration
- **Frazer** on Discord suggested the idea by stating "you can just combine them."
- This project takes inspiration from:
  - [st-moe-pytorch](https://github.com/lucidrains/st-moe-pytorch)
  - [efficient-kan](https://github.com/Blealtan/efficient-kan)

## Structure
- **src/moe_kan.py**: Contains the MoE implementation with KAN experts.
- **src/adaptive_gate.py**: Contains the implementation of the adaptive gate.
- **src/efficient_kan/kan.py**: Contains the efficient KAN implementation.
- **src/st-moe-pytorch/**: Contains the implementation of st-moe-pytorch.

## Installation
To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Usage
### Training
To train the model, use the following command:
```bash
python train.py --config configs/train_config.yaml
```
This will start the training process based on the configurations specified in `train_config.yaml`.

### Inference
To run inference, use:
```bash
python inference.py --config configs/inference_config.yaml --model_path path/to/model --input_path path/to/input
```
This will generate predictions using the trained model.

## Examples
Here is a basic example to get started:
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

print(output_data.shape)  # Should be [32, output_dim]
```

## Diagrams
### MoE with KAN Architecture
```plaintext
+----------------------+
|      Input           |
+---------+------------+
          |
          v
+---------+------------+
| Gate (Routing Layer) |
+---------+------------+
          |
  +-------+--------+
  |       |        |
  v       v        v
+---+   +---+    +---+
| E |   | E |    | E |
| x |   | x |    | x |
| p |   | p |    | p |
| 1 |   | 2 |    | 3 |
+---+   +---+    +---+
  |       |        |
  +-------+--------+
          |
          v
+---------+------------+
|      Output          |
+----------------------+
```

### Training Flow
```plaintext
+--------------------+
| Load Data          |
+---------+----------+
          |
          v
+---------+----------+
| Initialize Model   |
+---------+----------+
          |
          v
+---------+----------+
| Forward Pass       |
+---------+----------+
          |
          v
+---------+----------+
| Backward Pass      |
+---------+----------+
          |
          v
+---------+----------+
| Update Parameters  |
+--------------------+
```

## Expand Features
Implement additional functionalities like:
- **Dynamic Routing**: Implement a more sophisticated routing mechanism to select the most appropriate expert.
- **Advanced Logging**: Integrate advanced logging for better monitoring and debugging.
- **Visualization Tools**: Add visualization tools to better understand model performance and decision-making processes.
- **Hyperparameter Optimization**: Implement tools like Optuna for automated hyperparameter tuning.
- **Real-time Inference**: Enable real-time inference capabilities for live data processing.

## Contributing
Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Credits
- [st-moe-pytorch](https://github.com/lucidrains/st-moe-pytorch)
- [efficient-kan](https://github.com/Blealtan/efficient-kan)
- Idea inspired by Frazer on Discord

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.
```