"""
Written by: Samavedam Manikhanta Praphul
Version: 1.0
Description: 
This file has additional explorations done as part of extensions
This file will be less flexible
"""
#Thirdparty imports
import logging
import torch
from torch.optim import SGD, Adam, Adagrad, Adadelta
from torch.nn import Dropout2d
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from cv2 import filter2D


#Local imports
from models import BaseNetwork, DeepNetwork1, DeepNetwork2
from explore_cnn import quick_train_and_analyse_model, plot_graphs
from utils import get_fashion_mnist_data_loaders, train_and_analyse_model

   
def quick_plotting(indices, train_accuracies, test_accuracies,
                   train_losses, test_losses, time_elapsed, prefix:str ="Exp vary"):
    """
    This function purely plots the accuracies, losses and time elapsed against the indices
    
    Args
    indices:list[int]            indices of the graph plot
    train_accuracies:list[float] accuracies on train data set at indices
    test_accuracies:list[float]  accuracies on test  data set at indices
    train_losses:list[float]     losses on train data set at indices
    test_losses:list[float]      losses on test data set at indices
    time_elapsed:list[float]     time elapsed along the indices
    prefix:str                   prefix added to the graphs when storing
    Returns
    None    
    """
    plt.figure(figsize=(10, 10))
    plt.title(f"{prefix} Train and Test Accuracy")
    plt.scatter(indices,train_accuracies, color='b', label="Train Accuracy", marker='*')
    plt.plot(indices, train_accuracies, color='b')
    plt.scatter(indices, test_accuracies, color='r', label="Test Accuracy", marker="+")
    plt.plot(indices, test_accuracies, color='r')
    plt.savefig(f"{prefix} Train and Test Accuracy.png")
    plt.clf()

    plt.figure(figsize=(10, 10))
    plt.title(f"{prefix} Train and Test Loss")
    plt.scatter(indices, train_losses, color='b', marker='*')
    plt.plot(indices, train_losses, color='b', label="Train Loss")
    plt.scatter(indices, test_losses, color='r', label="Test Loss", marker="+")
    plt.plot(indices, test_losses, color='b', label="Test Loss")
    plt.savefig(f"{prefix} Train and Test Loss.png")
    plt.clf()

    plt.figure(figsize=(10, 10))
    plt.title(f"{prefix} Train Time Elapses")
    plt.plot(indices, time_elapsed, color='b', label="Time elapse")
    plt.savefig(f"{prefix} Training time.png")
    plt.clf()

def experiment_with_batch_size():
    """This function experiments with different values of the batch sizes on Base network
     & stores the results."""
    batch_sizes = [32, 64, 128, 512]
    learning_rate = 0.01
    momentum = 0.5
    train_dl, test_dl = get_fashion_mnist_data_loaders()

    train_losses, test_losses, train_accuracies, test_accuracies, time_elapses = [], [], [], [], []

    indices = [indx for indx, _ in enumerate(batch_sizes)]
    for batch_size in batch_sizes:
        model = BaseNetwork()
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        (train_accuracy, train_error, test_accuracy, test_error
        , time_elapsed )= quick_train_and_analyse_model(model, train_dl, test_dl, optimizer, 25, 5, batch_size)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        train_losses.append(train_error)
        test_losses.append(test_error)
        time_elapses.append(time_elapsed)
    quick_plotting(batch_sizes,train_accuracies,test_accuracies,
                   train_losses,test_losses,time_elapses, "Exp vary batch size")

def experiment_with_drop_out_rate():
    """This function experiments with different values of the dropout rates on Base network
     & stores the results."""
    drop_rates = [x/10.0 for x in range(0, 11, 2)]
    learning_rate = 0.01
    momentum = 0.5
    train_dl, test_dl = get_fashion_mnist_data_loaders()

    train_losses, test_losses, train_accuracies, test_accuracies, time_elapses = [], [], [], [], []

    indices = [indx for indx, _ in enumerate(drop_rates)]
    for drop_rate in drop_rates:
        model = BaseNetwork()
        model.convolution_stack[4] = Dropout2d(p=drop_rate)
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        (train_accuracy, train_error, test_accuracy, test_error
        , time_elapsed )= quick_train_and_analyse_model(model, train_dl, test_dl,optimizer, 25, 5, 64)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        train_losses.append(train_error)
        test_losses.append(test_error)
        time_elapses.append(time_elapsed)
        print()

    quick_plotting(indices,train_accuracies,test_accuracies,train_losses,test_losses,time_elapses,"Exp vary drop rates")

def visualize_first_layer_of_alex_net():
    """This prints the weights in the layer, the shape and size of filters in layer provided.
    Params
    layer_number:int the layer whose weights details have to be printed and return
    model: Module neural network model which needs to be analysed. 
    Returns specific layer weights
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    layer_number = 1
    model.eval()
    n_layer_weights = model.features[0].weight
    filter_count = n_layer_weights.shape[0]

    logging.basicConfig(filename='results/model_weights.log', encoding='utf-8', level=logging.INFO)

    logging.info(f"Number of filters in layer {layer_number} are {filter_count}")
    print(f"Number of filters in layer {layer_number} are {filter_count}")
    for filtr in range(filter_count):
        filter_weights = n_layer_weights[filtr,0]
        print(f"Filter {filtr + 1} of shape {filter_weights.shape} has weights \n{filter_weights}")
        logging.info(f"Filter {filtr + 1} of shape {filter_weights.shape} has weights \n{filter_weights}")
    visualize_filters(n_layer_weights=n_layer_weights, layer_number=layer_number)

    image = Image.open("data/alex_net_data/image.jpg", mode="r")
    resized_image = image.resize((224, 224))  # Resizing to fit the shape of dimensions.
    resized_image = resized_image.convert(
        mode="L"
    )  # Converting to black and white image.
    tensor = transforms.PILToTensor()(resized_image)  # Convert to Tensor
    tensor = tensor.to(dtype=torch.float)
    tensor.unsqueeze_(dim=0)  # To makeup for the batch sample

    filter_understanding(n_layer_weights, tensor, 1)


def visualize_filters(n_layer_weights: torch.Tensor, layer_number:int = 0) -> None:
    """This function visualizes the filters of the weights provided.
    Params
    n_layer_weights: torch.tensor tensor of the weights of the layer to visualize
    layer_number:int number of the layer which needs to be visualized. 
    """
    rows = 4
    columns = 8
    filters_count = n_layer_weights.shape[0]
    plt.figure(figsize=(16, 24))
    plt.suptitle(f"Visualizing filters learned in layer '{layer_number}' part 1")
    plt.xticks = []
    plt.yticks = []
    for fltr in range(1, (filters_count//2)+1, 1):
        plt.subplot(rows, columns, fltr)
        fltr_weight = n_layer_weights[fltr-1, 0]
        plt.imshow(fltr_weight.detach().squeeze(), cmap='viridis')
        plt.title(f"Filter {fltr}",fontsize=10)
        plt.axis("off")
    plt.savefig("results/AlexNet_layer_0_filters_1.png")
    plt.clf()

    plt.figure(figsize=(16, 24))
    plt.suptitle(f"Visualizing filters learned in layer '{layer_number}' part 2")
    plt.xticks = []
    plt.yticks = []
    for fltr in range(1, (filters_count//2)+1, 1):
        plt.subplot(rows, columns, fltr)
        indx = (filters_count//2) + fltr-1
        fltr_weight = n_layer_weights[indx, 0]
        plt.imshow(fltr_weight.detach().squeeze(), cmap='viridis')
        plt.title(f"Filter {indx+1}",fontsize=10)
        plt.axis("off")
    plt.savefig("results/AlexNet_layer_0_filters_2.png")
    plt.clf()
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
    rows = 8
    columns  = 16
    plt.figure(figsize=(20, 16))
    plt.suptitle(f"Information learned in layer {layer_number} of Alex Net")
    plt.xticks = []
    plt.yticks = []
    with torch.no_grad():
        for fltr in range(1, filter_count+1, 1):
            plt.subplot(rows, columns, (fltr-1)*2 + 1)
            plt.axis("off")
            plt.imshow(n_layer_weights[fltr-1, 0].detach().squeeze(), cmap='viridis')
            plt.title(f"Filter {fltr}", fontsize=6)
            kernel = n_layer_weights[fltr-1,0].detach().numpy()
            source = image.squeeze().detach().numpy()
            filtered_image = filter2D(src=source, ddepth = -1, kernel=kernel)
            plt.subplot(rows, columns, ((fltr-1)*2) + 2)
            plt.title(f"Info in filter {fltr}", fontsize=6)
            plt.imshow(filtered_image,cmap="gray")
            plt.axis("off")
    plt.savefig(f"results/Alex Net Information learned in layer {layer_number}.png")
    # plt.show()
    return None


def experiment_with_optimizers():
    """This function experiments with different values of the batch sizes on Base network
     & stores the results."""
    model = BaseNetwork()
    lr = 0.01
    mom = 0.5
    optimizers = [SGD(model.parameters(),lr,mom),
                  Adam(model.parameters(),lr),
                  Adagrad(model.parameters(),lr),
                  Adadelta(model.parameters(),lr)]
    train_dl, test_dl = get_fashion_mnist_data_loaders()

    indices = [indx for indx, _ in enumerate(optimizers)]
    for indx in indices:
        model = BaseNetwork()
        (epoch_indices, tr_losses_epochs, ts_losses_epochs, tr_acc_epochs, ts_acc_epochs,
            time_elapses, total_time, continual_train_indices,
            continual_train_losses, continual_test_indices) = train_and_analyse_model(model,train_dl,
                                                                 test_dl,optimizers[indx])

    plot_graphs(epoch_indices, tr_losses_epochs, ts_losses_epochs, tr_acc_epochs, ts_acc_epochs,
            time_elapses, total_time, continual_train_indices,
            continual_train_losses, continual_test_indices, prefix="Exp vary optimizer")

def experiment_with_layer_depth( log_interval: int = None, number_of_epochs: int = 1, train_batch_size: int = 64):
    """This function experiments with different number of layer depths"""
    neural_networks = [BaseNetwork, DeepNetwork1, DeepNetwork2]
    neural_net_names = ["Basic", "Basic+1layer", "Basic+2layers"]
    train_colors = ['b','c','g']
    test_colors = ['r','m','y']
    plt.figure(figsize=(20, 20))
    plt.suptitle("Exp Vary convolution layer depth Training and Testing Accuracies vs Epochs")

    train_data_loader, test_data_loader = get_fashion_mnist_data_loaders()
    learning_rate = 0.01
    momentum = 0.5

    for ind,neural_network in enumerate(neural_networks):
        model = neural_network()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        (epoch_indices, tr_losses_epochs, ts_losses_epochs, tr_acc_epochs, ts_acc_epochs,
            time_elapses, total_time, c_tr_indices,
            c_tr_losses, c_ts_indices) = train_and_analyse_model(model,train_data_loader,test_data_loader,optimizer,
                                log_interval,number_of_epochs,train_batch_size)
        plot_graphs(epoch_indices,tr_losses_epochs,ts_losses_epochs,tr_acc_epochs,
                    ts_acc_epochs,time_elapses,total_time,c_tr_indices,c_tr_losses,
                    c_ts_indices,prefix=f"Exp Vary Conv layers{neural_net_names[ind]}")

def main():
    """This is the function which runs when run as a standalone script.
    Returns 0 if the script exits successfully.
    """
    # experiment_with_drop_out_rate()
    experiment_with_batch_size()
    # visualize_first_layer_of_alex_net()
    # experiment_with_optimizers()
    # experiment_with_layer_depth()
    return 0

if __name__ == "__main__":
    main()


