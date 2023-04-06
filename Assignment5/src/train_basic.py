"""      
Written by: Samavedam Manikhanta Praphul
Version: 1.0
Description: 
This file trains a CNN architecture based model for 
MNIST dataset based on the system arguments passed. 
"""

# Importing required packages
# Third party imports
import torch # For PyTorch functionalities
from torchviz import make_dot

# Local imports
from models import BaseNetwork
from utils import parse_arguments, test_network, train_network, visualize_errors_over_training
from utils import get_mnist_data_loaders, visualize_data_loader_data


def main():
    """This is the function which runs when run as a standalone script.
    Returns 0 if the script exits successfully.
    """

    # Parse the command line arguments for get defaults.
    desc = "Trains basic neural network on MNIST for 5 epochs overwriting defaults by command line"
    samples, learning_rate, momentum, log_interval, train_batch_size,\
        test_batch_size, number_of_epochs = parse_arguments(description=desc)

    # Disable the cudnn
    torch.backends.cudnn.enabled = False

    # Setting the seed for reproducibility of results.
    random_seed = 45
    torch.manual_seed(random_seed)

    # Get the MNIST data if data doesn't exist
    train_data_loader, test_data_loader = get_mnist_data_loaders(train_batch_size=train_batch_size,
                                                                 test_batch_size=test_batch_size)

    #Visualize the data
    visualize_data_loader_data(train_data_loader, samples, "Visualizing first 8 train data points")
    visualize_data_loader_data(test_data_loader, samples, "Visualizing first 8 test data points")

    # Define the model, optimizer and loss function
    model = BaseNetwork()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=momentum)

    #Visualize the model
    for _, (image_data, _) in enumerate(train_data_loader):
        yhat = model(image_data)
        make_dot(yhat, params=dict(model.named_parameters())).render("base_network",format="png")
        break

    # Placeholders of training loss, testing loss along with indices
    train_losses = []
    train_indices = []
    test_losses = []
    test_indices = [epoch*len(train_data_loader.dataset) for epoch in range(number_of_epochs+1)]

    # Test error without training the model
    test_loss, _ = test_network(model=model,test_data_loader=test_data_loader)
    test_losses.append(test_loss)

    #Train the network for number of epochs
    for epoch in range(1, number_of_epochs+1, 1):
        losses, counter = train_network(model=model, train_data_loader=train_data_loader,
                                        optimizer=optimizer, log_interval = log_interval,
                                        epoch = epoch, batch_size=train_batch_size)
        train_losses.extend(losses)
        train_indices.extend(counter)
        test_loss, _ = test_network(model=model,test_data_loader=test_data_loader)
        test_losses.append(test_loss)

    # Visualize the training of the model over epochs
    visualize_errors_over_training(train_idx=train_indices, train_errors=train_losses, test_idx=test_indices, test_errors=test_losses)

    # Store the model
    torch.save(model.state_dict(), 'models/final_model.pth')

    # Standard exit status is 0
    return 0

if __name__ == "__main__":
    main()
    