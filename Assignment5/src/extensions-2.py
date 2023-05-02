"""      
Written by: Samavedam Manikhanta Praphul
Version: 1.0
Description: 
This file trains explore many aspects of a CNN architecture based model.
"""

# Importing required packages
# Third party imports
import torch # For PyTorch functionalities
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Local imports
from models import BaseNetwork, DeepNetwork1, DeepNetwork2, ManyParallelFiltersNetWork
from utils import (get_fashion_mnist_data_loaders, 
                   visualize_data_loader_data, train_and_analyse_model)

def experiment_with_different_filter_sizes(neural_network: torch.nn.Module, train_data_loader: torch.utils.data.DataLoader,
                                            test_data_loader: torch.utils.data.DataLoader, learning_rate:float, momentum:float,
                                            log_interval: int = None, number_of_epochs: int = 1, train_batch_size: int = 64):

    """This function experiments with different valid/possible filter sizes for FashionMNIST
    As FashionMNIST areof sizes 28 x28x1, only filter sizes possible of Base Network are 3,5,7
    """
   
    filtersizes = [1, 5, 9]
    fc_layer_values = [980, 320, 20]
    train_colors = ['b','c','g']
    test_colors = ['r','m','y']
    plt.figure(figsize=(20, 20))
    plt.suptitle("Exp Vary Filters size Training and Testing Accuracies vs Epochs")
    # plt.suptitle("Exp Vary Filters size Training and Testing Losses vs Epochs")
    for indx, kernel_size in enumerate(filtersizes):
        
        model = neural_network()

        print("Base Network:")
        print(model)
        # Modify the model to have the desired filter
        for layer in model.convolution_stack:
            if isinstance(layer, torch.nn.Conv2d):
                layer.kernel_size=(kernel_size, kernel_size)
        model.classification_stack[0].in_features = fc_layer_values[indx]
        print(f"Modified model wth filter size {kernel_size}")
        print(model)

        # Setting up the optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=momentum)
        # Train the modified model
        (epoch_indices, tr_losses_epochs, ts_losses_epochs, tr_acc_epochs, ts_acc_epochs,
            time_elapses, total_time, continual_train_indices,
            continual_train_losses, continual_test_indices) = train_and_analyse_model(model,train_data_loader,test_data_loader,optimizer,
                                log_interval,number_of_epochs,train_batch_size)
    
        plt.plot(epoch_indices, tr_acc_epochs, color=train_colors[indx], label=f'{kernel_size}_Train accuracies')
        plt.plot(epoch_indices, tr_acc_epochs, f'{train_colors[indx]}*')
        plt.plot(epoch_indices, ts_acc_epochs, color=test_colors[indx], label=f"{kernel_size}_Test accuracies")
        plt.plot(epoch_indices, ts_acc_epochs, f'{test_colors[indx]}*')

        # plt.plot(epoch_indices, tr_losses_epochs, train_colors[indx], label=f"{kernel_size}_Train error")
        # plt.plot(epoch_indices, tr_losses_epochs, f'{train_colors[indx]}^')
        # plt.plot(epoch_indices, ts_losses_epochs,  test_colors[indx], label=f"{kernel_size}_Test error")
        # plt.plot(epoch_indices, ts_losses_epochs,  f'{test_colors[indx]}*')
        
    plt.legend()
    # plt.savefig("Exp Vary Filters size Training and Testing Accuracies vs Epochs.png")
    plt.savefig("Exp Vary Filters size Training and Testing Losses vs Epochs.png")
    plt.show()

def experiment_with_epochs(neural_network,train_data_loader,test_data_loader,
                           learning_rate, momentum,log_interval,epoch_count,train_batch_size):
    """This function does the experiment with vary large value o
    f epochs 
    and stores the images while displaying to user"""
    model = neural_network()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=momentum)
    (epoch_indices, tr_losses_epochs, ts_losses_epochs, tr_acc_epochs, ts_acc_epochs,
            time_elapses, total_time, continual_train_indices,
            continual_train_losses, continual_test_indices) = train_and_analyse_model(model,train_data_loader,
                                                                 test_data_loader,optimizer,
                                                                 log_interval,epoch_count,
                                                                 train_batch_size)
    
    plot_graphs(epoch_indices, tr_losses_epochs, ts_losses_epochs, tr_acc_epochs, ts_acc_epochs,
            time_elapses, total_time, continual_train_indices,
            continual_train_losses, continual_test_indices, prefix="Exp Vary Epochs")


def experiment_with_layer_depth(train_data_loader:DataLoader, test_data_loader: DataLoader, learning_rate:float, momentum:float,
                                            log_interval: int = None, number_of_epochs: int = 1, train_batch_size: int = 64):
    neural_networks = [BaseNetwork, DeepNetwork1, DeepNetwork2]
    neural_net_names = ["Basic", "Basic+1layer", "Basic+2layers"]
    train_colors = ['b','c','g']
    test_colors = ['r','m','y']
    plt.figure(figsize=(20, 20))
    plt.suptitle("Exp Vary convolution layer depth Training and Testing Accuracies vs Epochs")
    for neural_network in neural_networks:
        model = neural_network()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        (epoch_indices, tr_losses_epochs, ts_losses_epochs, tr_acc_epochs, ts_acc_epochs,
            time_elapses, total_time, continual_train_indices,
            continual_train_losses, continual_test_indices) = train_and_analyse_model(model,train_data_loader,test_data_loader,optimizer,
                                log_interval,number_of_epochs,train_batch_size)
        # ToDo here
    pass
        





    pass

def plot_graphs(epoch_indices, tr_losses_epochs, ts_losses_epochs, tr_acc_epochs, ts_acc_epochs,
            time_elapses, total_time, continual_train_indices,
            continual_train_losses, continual_test_indices, prefix:str = "Exp abc"):
    """This function plots and stores the plots of 
    1. Training and Testing loss against #samples seen
    2. Training and Testing Accuracies vs Epochs
    3. Time taken to train vs Epochs
    4. Training and Testing Losses vs Epochs
    """

    # Create a PdfPages object
    pdf = PdfPages(f'{prefix}.pdf')

    figure1 = plt.figure(figsize=(20, 20))
    plt.plot(continual_train_indices, continual_train_losses, color='b', label='Train loss')
    plt.scatter(continual_test_indices, ts_losses_epochs, color= 'r', marker='^')
    plt.plot(continual_test_indices, ts_losses_epochs, color= 'r', label="Test loss")
    plt.title(f"{prefix} Training and Testing loss vs examples seen")
    plt.legend()
    pdf.savefig(figure1)
    plt.savefig(f"{prefix} Training and Testing loss vs examples seen.png")
    # plt.show()

    figure2= plt.figure(figsize=(20, 20))
    print(f"Time taken: {total_time}")
    plt.title(f"{prefix} Time taken to train vs Epochs")
    plt.plot([x for x in range(len(time_elapses))], time_elapses)
    plt.xlabel("Epochs")
    plt.ylabel("Training time vs Epochs")
    pdf.savefig(figure2)
    plt.legend()
    plt.savefig(f"{prefix} Time taken to train vs Epochs.png")
    # plt.show()

    figure3= plt.figure(figsize=(20, 20))
    plt.plot(epoch_indices, tr_acc_epochs, color='b', label='Train accuracy')
    plt.plot(epoch_indices, tr_acc_epochs, 'b*')
    plt.plot(epoch_indices, ts_acc_epochs, color='r', label="Test accuracy")
    plt.plot(epoch_indices, ts_acc_epochs, 'rp')
    plt.title(f"{prefix} Training and Testing Accuracies vs Epochs")
    plt.legend()
    plt.savefig(f"{prefix} Training and Testing Accuracies vs Epochs.png")
    # plt.show()
    pdf.savefig(figure3)

    figure4 = plt.figure(figsize=(20, 20))
    plt.plot(epoch_indices, tr_losses_epochs, 'b', label="Train accuracies")
    plt.plot(epoch_indices, tr_losses_epochs, 'b^')
    plt.plot(epoch_indices, ts_losses_epochs,  'r', label="Test accuracies")
    plt.plot(epoch_indices, ts_losses_epochs,  'r*')
    plt.title(f"{prefix} Training and Testing Losses vs Epochs")
    plt.legend()
    plt.savefig(f"{prefix} Training and Testing Losses vs Epochs.png")
    # plt.show()
    pdf.savefig(figure4)

    pdf.close()

def experiment_with_multiple_parallel_filters(train_data_loader:DataLoader, test_data_loader:DataLoader, learning_rate:float, momentum:float,
                                            log_interval: int = None, number_of_epochs: int = 1, train_batch_size: int = 64):
    """This function experiments by varying the number of filters being used at convolution layer
    """
    plt.figure(figsize=(20, 20))
    plt.suptitle("Exp Vary #Filters Training and Testing Accuracies vs Epochs")

    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []
    filter_counts = [x for x in range(5, 30, 5)]
    for filter_count in filter_counts:
        model = ManyParallelFiltersNetWork(filter_count)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        # Train the model
        (_, tr_losses_epochs, ts_losses_epochs, tr_acc_epochs, ts_acc_epochs,
               _, _, _,
                _, _) = train_and_analyse_model(model, train_data_loader,
                                                                    test_data_loader,optimizer,
                                                                    log_interval, number_of_epochs,
                                                                    train_batch_size)
        # Plot the model metrics over time
        train_accuracy.append(tr_acc_epochs[-1])
        test_accuracy.append(ts_acc_epochs[-1])
        train_losses.append(tr_losses_epochs[-1])
        test_losses.append(ts_losses_epochs[-1])
    plt.figure(figsize=(10, 10))
    plt.scatter(filter_counts,train_accuracy, color='b', label="Train Accuracy", marker='*')
    plt.plot(filter_counts, train_accuracy, color='r')
    plt.scatter(filter_counts, test_accuracy, color='r', label="Test Accuracy", marker="+")
    plt.savefig("Exp Vary Filter Count Train and Test Accuracy.png")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.scatter(filter_counts, train_losses, color='b', label="Train Loss", marker='*')
    plt.scatter(filter_counts, test_losses, color='r', label="Test Loss", marker="+")
    plt.savefig("Exp Vary Filter Count Train and Test Losses.png")
    plt.show()




def main():
    """This is the function which runs when run as a standalone script.
    Returns 0 if the script exits successfully.
    """
    # Default values
    train_batch_size = 64
    test_batch_size = 1000
    samples = 10
    learning_rate= 0.01
    momentum = 0.5
    number_of_epochs = 5
    log_interval = 10


    # Disable the cudnn
    torch.backends.cudnn.enabled = False

    # Setting the seed for reproducibility of results.
    random_seed = 45
    torch.manual_seed(random_seed)

    # Get the MNIST data if data doesn't exist
    train_data_loader, test_data_loader = get_fashion_mnist_data_loaders(train_batch_size=train_batch_size,
                                                                 test_batch_size=test_batch_size)

    #Visualize the data
    # visualize_data_loader_data(train_data_loader, samples, "Visualizing first 8 train data points")
    # visualize_data_loader_data(test_data_loader, samples, "Visualizing first 8 test data points")

    # Define the model, optimizer and loss function
    # optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=momentum)

    # Experimentation with filter sizes with rest as constant
    # Expectation: Is that the accuracy increases upto certain filter size, later there will not be much improvement. 
    
    # experiment_with_different_filter_sizes(BaseNetwork,train_data_loader,test_data_loader,learning_rate,
    #                                        momentum,log_interval,number_of_epochs,train_batch_size)

    # Experimentation with epochs with rest as constant
    # Expectation: Loss decreases as Epochs increase upto certain limit.)
    # model = BaseNetwork
    # epoch_count = 20
    # experiment_with_epochs(model, train_data_loader, test_data_loader, learning_rate,momentum,
    #                        log_interval,epoch_count,train_batch_size)

    # Experimentation with number of filters
    experiment_with_multiple_parallel_filters(train_data_loader, test_data_loader,
                                              learning_rate, momentum,log_interval,
                                              number_of_epochs, train_batch_size)

    # Standard exit status is 0
    return 0

if __name__ == "__main__":
    main()
    