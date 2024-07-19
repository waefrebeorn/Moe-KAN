import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import yaml
import argparse
from src.moe_kan import MoE

def load_data(batch_size):
    # Example data: replace with actual data loading logic
    x = torch.randn(1000, 128)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    input_dim = config['model']['input_dim']
    num_experts = config['model']['num_experts']
    num_epochs = config['training']['num_epochs']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']

    model = MoE(input_dim, num_experts)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataloader = load_data(batch_size)

    train(model, dataloader, criterion, optimizer, num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Moe-KAN model")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    main(args.config)
