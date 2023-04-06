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

# Local imports
from test_basic import load_model
from models import BaseNetwork
from utils import freeze_layers_and_modify_last_layer, train_and_plot_accuracies


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
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    # Train and plot the accuracy based on the
    train_and_plot_accuracies(model=model, epochs=epochs, optimizer=optimizer,
                              train_data_loader=greek_train, log_interval=log_interval,
                              batch_size=batch_size,model_path="models/model_greek.pth",
                              optimizer_path="models/optim_greek.pth")

    # Save the trained model
    torch.save(model.state_dict(),"models/model_greek.pth")

if __name__ == "__main__":
    main()
    