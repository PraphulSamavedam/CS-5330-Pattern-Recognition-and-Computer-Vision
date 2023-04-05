"""      
Written by: Samavedam Manikhanta Praphul
Version: 1.0
Description: 
This file trains a CNN architecture based model for 
MNIST dataset based on the system arguments passed. 
"""

# Importing required packages
import sys # For accessing the command line arguments passed
import argparse # For parsing the argument passed to file as script.
from torch import nn # For custom neural network
import torch # For PyTorch functionalities
from torch.utils.data import DataLoader # Access the data from the downloaded datasets
from torch.nn import functional as F
import matplotlib.pyplot as plt

from utils import get_mnist_data_loaders, visualize_data_loader_data

# class definitions
class MyNetwork(nn.Module):
    '''
    This neural network model represents the neural network model similar to LeNet-5 developed by LeCunn in 1998
    tuned to predict the class out of 10 possible classes by processing 28 x 28 grayscale images.
    Significant changes with respect to LeNet-5 are as follows:
    1. Activation is ReLU instead of Sigmoid function. 
    2. All convolutions are valid convolutions i.e. padding = 0.
    There shall be other minor changes as well. 
    '''

    def __init__(self):
        """This initializes my network with the required parameters which needs to be learned."""
        # Convolution stack
        super(MyNetwork,self).__init__()
        self.convolution_stack = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5,5)) # Output shape ( 10, 24, 24)
                                               ,nn.MaxPool2d(kernel_size=2, stride = 2, padding=0) # Output shape (10, 12, 12)
                                               ,nn.ReLU()
                                               ,nn.Conv2d(in_channels=10, out_channels=20, kernel_size = 5) # Output shape (20, 8, 8)
                                               ,nn.Dropout2d(p=0.5)
                                               ,nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # Output shape (20, 4, 4 )
                                               ,nn.ReLU()
                                               ,nn.Flatten(start_dim=1, end_dim=-1) # Output shape 4*4*20 = 320
                                              )
        # Classification stack
        self.classification_stack = nn.Sequential(nn.Linear(in_features=320, out_features=50) 
                                                  ,nn.ReLU()
                                                  ,nn.Linear(in_features=50, out_features=10)
                                                  ,nn.LogSoftmax(dim=None)
                                                 )
        self.optim = None
        self.loss_function = None

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, data):
        """ This function computes a forward pass through the network."""
        encoding = self.convolution_stack(data)
        return self.classification_stack(encoding)

def train_network(model:nn.Module, train_data_loader: DataLoader, optimizer: torch.optim
                  ,log_interval:int = None, epoch: int  = 1, batch_size:int = 64,
                  model_path:str = "models/base_model.pth", optim_path:str = "models/base_optimizer.pth"):
    """This function trains the neural network module passed on the 
    train_dataset for single epoch and returns the training error at 
    regular intervals while saving the latest model if log_interval is not None

    Params:
    model: nn.Module model which needs to be trained. 
    train_data_loader : DataLoader data loader corresponding to the loading data
    optimizer: torch.optim optimizer based on which the model has to take the steps during optimization
    log_interval: int default = None
    epoch: int Current epoch value

    Returns:
    losses, indices
    """
    model.train()
    losses = []
    counter = []
    for batch_idx, (image_data, image_labels) in enumerate(train_data_loader):
        predictions  = model(image_data)
        loss = F.nll_loss(predictions, image_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if log_interval is not None:
                if batch_idx % log_interval == 0:
                    print(f"Train error: Epoch {epoch},[{batch_idx * len(image_data)}/{len(train_data_loader.dataset)}", end="\t")
                    print(f"({100*batch_idx / len(train_data_loader):.2f}%)]", end="")
                    print(f"Loss: {loss.item():.06f}")
                    losses.append(loss.item())
                    counter.append((batch_idx*batch_size) + ((epoch-1)* len(train_data_loader.dataset)))
                    torch.save(model.state_dict(), model_path)
                    torch.save(optimizer.state_dict(),optim_path)
    return losses, counter

def test_network(model:nn.Module, test_data_loader:DataLoader):
    """This function calculates the performance on test data set provided
    Returns loss and accuracy of model"""
    model.eval() # To set the model in evaluation mode -> No training
    test_loss:float = 0
    correct_predictions:int = 0
    with torch.no_grad():
        for data, target in test_data_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            prediction = output.data.max(1, keepdim=True)[1]
            correct_predictions += prediction.eq(target.data.view_as(prediction)).sum()
        # Average the loss over all the samples
        test_loss /= len(test_data_loader.dataset)
        accuracy = 100. * correct_predictions / len(test_data_loader.dataset)
        print(f"\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct_predictions}/{len(test_data_loader.dataset)} ({accuracy:.2f}%)\n")
    return test_loss, accuracy

def visualize_errors_over_training(train_idx, train_errors, test_idx, test_errors):
    """This function visualizes the data over time"""
    plt.figure(figsize=(10, 10))
    plt.plot(train_idx, train_errors, c='b', label="Training loss")
    plt.scatter(test_idx, test_errors, c='r', marker='o', label="Testing loss")
    plt.legend()
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")
    plt.title("Training and Testing Loss vs number of examples seen")
    plt.savefig('Training and Testing Loss vs number of examples seen')
    plt.show()

def main(argv):
    """This function handles the command line arguments passed &
    runs the script as per command line argmuments 
    Params
    argv command line arguments passed to the file in script mode.
    """
    # Parse the command line argument passed.
    parser = argparse.ArgumentParser(prog="train.py",
                                     usage="main.py -s <samples> -r <learning_rate> -m <momentum> -l <logging>\
                                          -br <training_batch_size> -bs <testing_batch_size> -e <training #epochs>\
                                          Defaults\
                                            samples = 8, learning_rate = 0.01, momentum=0.5, logging = 10\
                                            training_batch_size = 64, testing_batch_size = 1000, epochs = 5",
                                     description="Trains the custom neural network for MNIST for 5 epochs overwrite defaults with command line arguments.",epilog="")
    parser.add_argument('-s','--samples',required=False, type=int)
    parser.add_argument('-r', '--rate',required=False, type=float)
    parser.add_argument('-m', '--momentum',required=False, type=float)
    parser.add_argument('-l', '--logging',required=False, type=int)
    parser.add_argument('-br', '--train_batch_size',required=False, type=int)
    parser.add_argument('-bs', '--test_batch_size',required=False, type=int)
    parser.add_argument('-e', '--epochs',required=False, type=int)
    args = parser.parse_args()
    momentum = args.momentum if args.momentum else 0.5
    learning_rate = args.rate if args.rate else 0.01
    samples = args.samples if args.samples else 8
    log_interval = args.logging if args.logging else 10
    train_batch_size = args.logging if args.train_batch_size else 64
    test_batch_size = args.logging if args.test_batch_size else 1000
    number_of_epochs = args.epochs if args.epochs else 5

    # Disable the cudnn 
    torch.backends.cudnn.enabled = False

    # Setting the seed for reproducibility of results.
    random_seed = 45
    torch.manual_seed(random_seed)

    # Get the MNIST data if data doesn't exist
    train_data_loader, test_data_loader = get_mnist_data_loaders(train_batch_size=train_batch_size, test_batch_size=test_batch_size)
    
    #Visualize the data 
    visualize_data_loader_data(train_data_loader, samples, "Visualizing first 8 train data points")
    visualize_data_loader_data(test_data_loader, samples, "Visualizing first 8 test data points")
    
    # Define the model, optimizer and loss function 
    model = MyNetwork()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=momentum)

    #Visualize the model
    print(model)

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
    main(sys.argv)
    