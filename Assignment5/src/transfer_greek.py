"""
Written by: Samavedam Manikhanta Praphul
Version: 1.0
Description: 
This file transfer learns from trained model to recognize 
the Greek letters alpha, beta, gamma
"""

# Thirdparty imports
import torch
import torchvision
from torchviz import make_dot

# Local imports
from test_basic import load_model
from models import BaseNetwork
from utils import freeze_layers_and_modify_last_layer, train_and_plot_accuracies
from transforms import GreekTransform


def main():
    """This is the function which runs when run as a standalone script.
    Returns
    0 if the script exits successfully."""
    epochs = 15
    learning_rate = 0.03
    momentum = 0.1
    log_interval = 3
    batch_size = 5

    # Disable the cudnn
    torch.backends.cudnn.enabled = False

    # Setting the seed for reproducibility of results.
    random_seed = 45
    torch.manual_seed(random_seed)

    # Load the model from file
    model = load_model(BaseNetwork,model_path="models/final_model.pth")
    # Modify the model to freeze all layers & modify last FC layer to fit
    model = freeze_layers_and_modify_last_layer(model=model, output_features=3)

    # Train dataset provided
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(root="data/greek_train",
                                         transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            GreekTransform(),
                                            torchvision.transforms.Normalize((0.1307,),
                                                                             (0.3801,),)
                                         ])), batch_size = batch_size, shuffle=True )

    #Visualize the model
    for _, (image_data, _) in enumerate(greek_train):
        yhat = model(image_data)
        make_dot(yhat, params=dict(model.named_parameters())).render("greek_network",format="png")
        break

    # Optimizer for training the model
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    # Train and plot the accuracy for all epochs
    train_and_plot_accuracies(model=model, epochs=epochs, optimizer=optimizer,
                              train_data_loader=greek_train, log_interval=log_interval,
                              batch_size=batch_size,model_path="models/model_greek.pth",
                              optimizer_path="models/optim_greek.pth")

    # Save the trained model
    torch.save(model.state_dict(),"models/model_greek.pth")

if __name__ == "__main__":
    main()
    