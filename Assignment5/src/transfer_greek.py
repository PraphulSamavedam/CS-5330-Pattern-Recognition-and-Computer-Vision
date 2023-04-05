"""
Written by: Samavedam Manikhanta Praphul
Version: 1.0
Description: 
This file transfer learns from trained model to recognize 
the Greek letters alpha, beta, gamma
"""

from test import load_model
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from train import train_network
from train import test_network


class GreekTransform:
    """This class represents the transformation required to convert 133 x 133 color images of
     greek letters
     """
    def __init__(self) -> None:
        pass

    def __call__(self, x_input) :
        x_input = torchvision.transforms.functional.rgb_to_grayscale(x_input)
        x_input = torchvision.transforms.functional.affine(x_input, 0, (0, 0), 36/128, 0)
        x_input = torchvision.transforms.functional.center_crop(x_input, (28, 28))
        return torchvision.transforms.functional.invert(x_input)

def freeze_layers_and_modify_last_layer(model: nn.Module, output_features: int):
    """This function freezes all the layers prior to the last FC layer.
    
    Args
    model:Module neural network which needs to be freezed.
    """
    for param in model.parameters():
        param.requires_grad = False
    model.classification_stack[-2] = nn.Linear(in_features= 50,
                                               out_features = output_features,
                                               bias = True)
    return model


def main():
    """This is the function which runs when run as a standalone script.
    Returns
    0 if the script exits successfully."""

    epochs = 25
    learning_rate = 0.03
    momentum = 0.1
    log_interval = 5
    batch_size = 5

    # Load the model from file
    model = load_model(model_path="models/final_model.pth")
    # Modify the model to freeze all layers & modify last FC layer to fit
    model = freeze_layers_and_modify_last_layer(model=model, output_features=3)

    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(root="data/greek_train",
                                         transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            GreekTransform(),
                                            torchvision.transforms.Normalize((0.1307,),
                                                                             (0.3801,),)
                                         ])), batch_size = batch_size, shuffle=True )

    train_losses = []
    train_indices = []
    accuracies = []
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    for epoch in range(1, epochs+1, 1):
        losses, indices = train_network(model= model, train_data_loader= greek_train,
                    optimizer=optimizer, log_interval=log_interval,
                    epoch=epoch, batch_size=batch_size,
                    model_path="models/model_greek.pth", optim_path="models/optim_greek.pth")
        train_losses.extend(losses)
        train_indices.extend(indices)
        _, accuracy = test_network(model=model, test_data_loader=greek_train)
        accuracies.append(accuracy)
    plt.figure(figsize=(10, 10))
    plt.plot(train_indices, train_losses, c='g', label="Training loss")
    plt.legend()
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")
    plt.title("Training loss vs number of examples seen")
    plt.savefig("results/transfer_learning_errors.png")
    plt.show()

    epochs_indices = [x for x in range(epochs)]
    plt.plot(epochs_indices, accuracies,c='r',marker='x', label= "Train accuracy")
    plt.title("Training accuracy vs epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("results/transfer_learning_accuracies.png")
    plt.show()

if __name__ == "__main__":
    main()
    