import torch
from torch.utils.data import DataLoader, TensorDataset
import argparse
import yaml
from src.moe_kan import MoE

def load_data(input_path, batch_size):
    # Replace with actual data loading logic
    x = torch.load(input_path)  # Assuming input data is saved as a tensor
    dataset = TensorDataset(x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def inference(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x_batch in dataloader:
            outputs = model(x_batch[0])
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
    return predictions

def main(config_path, model_path, input_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    input_dim = config['model']['input_dim']
    num_experts = config['model']['num_experts']
    batch_size = config['inference']['batch_size']

    model = MoE(input_dim, num_experts)
    model.load_state_dict(torch.load(model_path))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = load_data(input_path, batch_size)
    predictions = inference(model, dataloader)
    print(predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with Moe-KAN model")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the input data")
    args = parser.parse_args()
    main(args.config, args.model_path, args.input_path)
