"""
Written by: Samavedam Manikhanta Praphul
Version: 1.0
Description:
This file visualizes the different filters in the model learned from several training examples
"""
# Third party imports
import matplotlib.pyplot as plt
from cv2 import filter2D
import torch
from torch.nn import Module
# Local imports
from models import BaseNetwork
from test_basic import load_model
from utils import get_mnist_data_loaders


def get_layer_weights(model:Module, layer_number:int = 0):
    """This prints the weights in the layer, the shape and size of filters in layer provided.
    Params
    layer_number:int the layer whose weights details have to be printed and return
    model: Module neural network model which needs to be analysed. 
    Returns specific layer weights
    """
    assert layer_number < 5 # To ensure that the layer corresponds to the convolution stack. 
    n_layer_weights = model.convolution_stack[layer_number].weight
    filter_count = n_layer_weights.shape[0]
    print(f"Number of filters in layer {layer_number} are {filter_count}")
    for filtr in range(filter_count):
        filter_weights = n_layer_weights[filtr,0]
        print(f"Filter {filtr + 1} of shape {filter_weights.shape} has weights \n{filter_weights}")
    return n_layer_weights

def visualize_filters(n_layer_weights: torch.Tensor, layer_number:int = 0) -> None:
    """This function visualizes the filters of the weights provided.
    Params
    n_layer_weights: torch.tensor tensor of the weights of the layer to visualize
    layer_number:int number of the layer which needs to be visualized. 
    Raises
    Assertion Error if more than 12 filters have to visualized
    """
    assert n_layer_weights.shape[0] <= 12
    rows = 3
    columns = 4
    filters_count = n_layer_weights.shape[0]
    plt.figure(figsize=(12, 12))
    plt.suptitle(f"Visualizing filters learned in layer '{layer_number}'")
    plt.xticks = []
    plt.yticks = []
    for fltr in range(1, filters_count+1, 1):
        plt.subplot(rows, columns, fltr)
        fltr_weight = n_layer_weights[fltr-1, 0]
        plt.imshow(fltr_weight.detach().squeeze(), cmap='viridis')
        plt.title(f"Filter {fltr}",fontsize=10)
        plt.axis("off")
    plt.savefig("results/layer_0_filters.png")
    plt.show()
    return None

def filter_understanding(n_layer_weights: torch.Tensor, image: torch.Tensor,
                         layer_number:int = 0):
    """This function applies the different filters on the image_provided & outputs the results
    Returns 
    None
    Params
    n_layer_weights: torch.Tensor tensor of the filters.
    image: torch.Tensor image on which filter impact
    layer_number: int the layer which is being analyzed. 
    """
    filter_count = n_layer_weights.shape[0]
    rows = 5
    columns  = 4
    plt.figure(figsize=(20, 16))
    plt.suptitle(f"Information learned in layer {layer_number}")
    plt.xticks = []
    plt.yticks = []
    with torch.no_grad():
        for fltr in range(1, filter_count+1, 1):
            plt.subplot(rows, columns, (fltr-1)*2 + 1)
            plt.axis("off")
            plt.imshow(n_layer_weights[fltr-1, 0].detach().squeeze(), cmap='viridis')
            plt.title(f"Filter {fltr}", fontsize=7)
            kernel = n_layer_weights[fltr-1,0].detach().numpy()
            source = image.detach().numpy()
            filtered_image = filter2D(src=source, ddepth = -1, kernel=kernel)
            plt.subplot(rows, columns, ((fltr-1)*2) + 2)
            plt.title(f"Learnt info in filter {fltr}", fontsize=7)
            plt.imshow(filtered_image,cmap="gray")
            plt.axis("off")
    plt.savefig(f"results/Information learned in layer {layer_number}.png")
    plt.show()
    return None

def main():
    """This is the function which runs when run as a standalone script.
    Returns
    0 if the script exits successfully."""
    model = load_model(BaseNetwork, "models/final_model.pth")
    n_layer_weights = get_layer_weights(model=model, layer_number=0)
    visualize_filters(n_layer_weights, layer_number=0)
    train_data_loader , _ = get_mnist_data_loaders()
    for data, _ in train_data_loader:
        image = data[0][0]
        break
    filter_understanding(n_layer_weights=n_layer_weights, image=image)
    return 0

if __name__ == "__main__":
    main()
    