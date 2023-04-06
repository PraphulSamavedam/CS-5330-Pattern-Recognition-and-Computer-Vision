"""
Written by: Samavedam Manikhanta Praphul
Version: 1.0
Description:
This file loads the model from a file and then runs it for some random images in test data 
"""

import random
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import get_mnist_data_loaders
from models import BaseNetwork


def predict(model: Module, test_data: DataLoader, count: int) -> None:
    """
    This function displays the prediction for the #count random samples from data loader provided.
    Params
    model:Module Trained neural network
    test_data:DataLoader dataloader of the data which needs to be randomly sampled.
    count:int number of random samples to be predicted.
    Returns None
    """

    model.eval()
    with torch.no_grad():
        for data, target in test_data:
            output = model(data)
            prediction = output.data.max(1, keepdim=True)[1]
            indices = [random.randint(0, len(target) - 1) for indx in range(count)]
            plt.figure(figsize=(20, 20))
            plt.axis("off")
            plt.title(f"Model predictions on {count} test samples")

            rows = 3
            columns = (
                (count // rows) + 1
                if count < rows * (count // rows)
                else (count // rows)
            )
            for index, indx in enumerate(indices):
                plt.subplot(rows, columns, index + 1)
                plt.imshow(data[indx].squeeze(), cmap="gray")
                plt.axis("off")
                plt.title(f"Prediction {prediction[indx].item()}")
            plt.savefig("results/Model predictions.png")
            plt.show()
            return None


def load_model(neural_network:Module, model_path: str) -> Module:
    """This function returns the model completely loaded from the file path provided.
    Returns
    nn.Module neural network which is loaded from the file path.
    """
    model = neural_network()
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)
    return model


def main():
    """This is the function which runs when run as a standalone script.
    Returns
    0 if the script exits successfully."""
    model = load_model(BaseNetwork, "models/final_model.pth")
    _, test_data = get_mnist_data_loaders()
    predict(model=model, test_data=test_data, count=9)
    return 0


if __name__ == "__main__":
    main()
