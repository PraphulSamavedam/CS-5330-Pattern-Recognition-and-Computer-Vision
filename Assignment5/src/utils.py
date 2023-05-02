"""
Written by: Samavedam Manikhanta Prahul
Version: 1.0
Description: This file has the utility functions that can be reused in other scripts. 
"""
import argparse  # For parsing the argument passed to file as script.
import time
import timeit
import torch
from torch.nn import (
    functional as F,
)
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import matplotlib.pyplot as plt

from torchvision.datasets import (MNIST, FashionMNIST)
from torchvision import transforms


def visualize_data_loader_data(
    data_loader: DataLoader, samples: int = 6, title: str = None):
    """This function visualizes the first n samples of the dataloader provided.
    Params
    data_loader: DataLoader data which needs to be visualized.
    samples:int = 6 number of samples of the data that you want to visualize.
    title:str = None, if title is None the data is presented with title
    'Visualizing first #samples data points'
    Raises
    Assertion Error if samples are not in range of [1, 10]
    """
    assert (samples >= 1) & (samples <= 10)
    fig = plt.figure(figsize=(10, 10))

    # Set the subplots configuration
    rows = 2
    cols = samples // rows
    if rows * cols < samples:
        cols += 1

    title = f"Visualizing first {samples} data points" if title == "" else title
    # Get the data to plot
    enumerator = enumerate(data_loader)
    _, (img_data, labels) = next(enumerator)
    plt.title(title)
    plt.axis("off")

    # Plotting the data
    for index in range(1, samples + 1):
        img = img_data[index]
        label = labels[index]
        fig.add_subplot(rows, cols, index)
        plt.title(f"Ground truth {label.item()}")
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")

    # Display the data
    plt.show()


def get_mnist_data_loaders(train_batch_size: int = 64, test_batch_size: int = 1000):
    """This function returns the data loaders corresponding to train_data and test_data"""
    train_data = MNIST(
        root="data/", download=True, train=True, transform=transforms.ToTensor()
    )
    mean_val = (train_data.train_data.detach().float().mean() / 255).item()
    stdv_val = (train_data.train_data.detach().float().std() / 255).item()
    training_data_loader = DataLoader(
        MNIST(
            root="data/",
            download=True,
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((mean_val,), (stdv_val,))]
            ),
        ),
        batch_size=train_batch_size,
    )

    testing_data_loader = DataLoader(
        MNIST(
            root="data/",
            download=True,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((mean_val,), (stdv_val,))]
            ),
        ),
        shuffle=False
        # Shuffle turned off to ensure reproducibility
        ,
        batch_size=test_batch_size,
    )

    return training_data_loader, testing_data_loader


def parse_arguments(description: str):
    """This function parses the command line arguments and returns the following arguments parsed

    samples, learning_rate, momentum, log_interval, train_bsize, test_bsize, epochs
    Defaults:
    samples = 8, learning_rate = 0.01, momentum=0.5, logging = 10
    training_batch_size = 64, testing_batch_size = 1000, epochs = 5
    """
    # Parse the command line argument passed.
    parser = argparse.ArgumentParser(
        prog=f"{__file__}",
        usage=f"""{__file__} -s <samples> -r <l_rate> -m <momentum>
                                     -l <logging>  -br <tr_batch_size> -bs <ts_batch_size> -e <epochs>
                                          Defaults
                                            samples = 8, learning_rate = 0.01, 
                                            momentum=0.5, logging = 10
                                            training_batch_size = 64, 
                                            testing_batch_size = 1000, epochs = 5""",
        description=f"{description}",
        epilog="",
    )
    parser.add_argument("-s", "--samples", required=False, type=int)
    parser.add_argument("-r", "--rate", required=False, type=float)
    parser.add_argument("-m", "--momentum", required=False, type=float)
    parser.add_argument("-l", "--logging", required=False, type=int)
    parser.add_argument("-br", "--train_batch_size", required=False, type=int)
    parser.add_argument("-bs", "--test_batch_size", required=False, type=int)
    parser.add_argument("-e", "--epochs", required=False, type=int)
    args = parser.parse_args()
    momentum = args.momentum if args.momentum else 0.5
    learning_rate = args.rate if args.rate else 0.01
    samples = args.samples if args.samples else 8
    log_interval = args.logging if args.logging else 10
    train_bsize = args.logging if args.train_batch_size else 64
    test_bsize = args.logging if args.test_batch_size else 1000
    epochs = args.epochs if args.epochs else 5

    dct = args.__dict__
    for key in dct:
        if dct[key] is not None:
            print(f"Received argument {key} as {dct[key]}")
    return (
        samples,
        learning_rate,
        momentum,
        log_interval,
        train_bsize,
        test_bsize,
        epochs,
    )


def train_network_single_epoch(
    model: nn.Module,
    train_data_loader: DataLoader,
    optimizer: optim,
    log_interval: int = None,
    epoch: int = 1,
    batch_size: int = 64,
    model_path: str = "models/base_model.pth",
    optim_path: str = "models/base_optimizer.pth",
):
    """This function trains the neural network module passed on the
    train_dataset for single epoch and returns the training error at
    regular intervals while saving the latest model if log_interval is not None
    Params:
    model: nn.Module model which needs to be trained.
    train_data_loader : DataLoader data loader corresponding to the loading data
    optimizer: torch.optim optimizer based on which model takes the steps during optimization
    log_interval: int default = None
    epoch: int Current epoch value
    Returns:
    losses, indices
    """
    model.train()
    losses = []
    counter = []
    for batch_idx, (image_data, image_labels) in enumerate(train_data_loader):
        predictions = model(image_data)
        loss = F.nll_loss(predictions, image_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if log_interval is not None:
            if batch_idx % log_interval == 0:
                print(
                    f"Train error: Epoch {epoch},[{batch_idx * len(image_data)}/{len(train_data_loader.dataset)}",
                    end="\t",
                )
                print(f"({100 * batch_idx / len(train_data_loader):.2f}%)]", end="")
                print(f"Loss: {loss.item():.06f}")
                losses.append(loss.item())
                counter.append(
                    (batch_idx * batch_size)
                    + ((epoch - 1) * len(train_data_loader.dataset))
                )
                torch.save(model.state_dict(), model_path)
                torch.save(optimizer.state_dict(), optim_path)
    return losses, counter


def test_network(model: nn.Module, test_data_loader: DataLoader):
    """This function calculates the performance on test data set provided
    Returns loss and accuracy of model"""
    model.eval()  # To set the model in evaluation mode -> No training
    test_loss: float = 0
    correct_predictions: int = 0
    with torch.no_grad():
        for data, target in test_data_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            prediction = output.data.max(1, keepdim=True)[1]
            correct_predictions += prediction.eq(target.data.view_as(prediction)).sum()
        # Average the loss over all the samples
        test_loss /= len(test_data_loader.dataset)
        accuracy = 100.0 * correct_predictions / len(test_data_loader.dataset)
        print(
            f"\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct_predictions}/{len(test_data_loader.dataset)} ({accuracy:.2f}%)\n"
        )
    return test_loss, accuracy


def train_and_plot_accuracies(
    model: nn.Module,
    epochs: int,
    optimizer: optim,
    train_data_loader: DataLoader,
    log_interval: int,
    batch_size: int,
    model_path: str = "models/model_transfer.pth",
    optimizer_path: str = "models/optim_transfer.pth",
    accuracy_results_path: str = "results/transfer_learning_accuracies.png",
    error_results_path: str = "results/transfer_learning_errors.png",
):
    """This function trains the network for the train data loader and plots
    the accuracies vs epochs and
    training losses vs training examples seen"""
    train_losses = []
    train_indices = []
    accuracies = []
    for epoch in range(1, epochs + 1, 1):
        losses, indices = train_network_single_epoch(
            model=model,
            train_data_loader=train_data_loader,
            optimizer=optimizer,
            log_interval=log_interval,
            epoch=epoch,
            batch_size=batch_size,
            model_path=model_path,
            optim_path=optimizer_path,
        )
        train_losses.extend(losses)
        train_indices.extend(indices)
        _, accuracy = test_network(model=model, test_data_loader=train_data_loader)
        accuracies.append(accuracy)
    plt.figure(figsize=(10, 10))
    plt.plot(train_indices, train_losses, c="g", label="Training loss")
    plt.legend()
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")
    plt.title("Training loss vs number of examples seen")
    plt.savefig(error_results_path)
    plt.show()

    epochs_indices = [x for x in range(epochs)]
    plt.plot(epochs_indices, accuracies, c="r", marker="x", label="Train accuracy")
    plt.title("Training accuracy vs epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(accuracy_results_path)
    plt.show()
    return None


def freeze_layers_and_modify_last_layer(model: nn.Module, output_features: int):
    """This function freezes all the layers prior to the last FC layer.
    Args
    model:Module neural network which needs to be freezed.
    """
    for param in model.parameters():
        param.requires_grad = False
    model.classification_stack[-2] = nn.Linear(
        in_features=50, out_features=output_features, bias=True
    )
    return model


def visualize_errors_over_training(
    train_idx, train_errors, test_idx, test_errors
) -> None:
    """This function visualizes the errors in training data over time
    Args
    train_idx : number of examples seen by model till then used as index
    train_errors: training errors at regular intervals
    test_idx  : number of examples seen by model till then used as index
    train_errors: testing  errors at regular intervals
    """
    plt.figure(figsize=(10, 10))
    plt.plot(train_idx, train_errors, c="b", label="Training loss")
    plt.scatter(test_idx, test_errors, c="r", marker="o", label="Testing loss")
    plt.legend()
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")
    plt.title("Training and Testing Loss vs number of examples seen")
    plt.savefig("Training and Testing Loss vs number of examples seen")
    plt.show()


def get_fashion_mnist_data_loaders(
    train_batch_size: int = 64, test_batch_size: int = 1000
):
    """This function returns the data loaders corresponding to train_data and test_data"""
    train_data = FashionMNIST(
        root="data/", download=True, train=True, transform=transforms.ToTensor()
    )
    mean_val = (train_data.train_data.detach().float().mean() / 255).item()
    stdv_val = (train_data.train_data.detach().float().std() / 255).item()
    training_data_loader = DataLoader(
        FashionMNIST(
            root="data/",
            download=True,
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((mean_val,), (stdv_val,))]
            ),
        ),
        batch_size=train_batch_size,
    )

    testing_data_loader = DataLoader(
        FashionMNIST(
            root="data/",
            download=True,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((mean_val,), (stdv_val,))]
            ),
        ),
        shuffle=False
        # Shuffle turned off to ensure reproducibility
        ,
        batch_size=test_batch_size,
    )

    return training_data_loader, testing_data_loader


def train_and_analyse_model(model: nn.Module,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    optimizer: optim,
    log_interval: int = None,
    number_of_epochs: int = 1,
    train_batch_size: int = 64):
    """This function trains the given model for specified epochs on trainset
    and provides the 
    (epoch_indices, tr_losses_epochs, ts_losses_epochs, tr_acc_epochs, ts_acc_epochs,
            time_elapses, time_elapsed, continual_train_indices, continual_train_losses, test_indices)"""
    start = timeit.timeit()

     # Placeholders of training loss, testing loss along with indices
    tr_losses_epochs = []
    ts_losses_epochs = []
    tr_acc_epochs = []
    ts_acc_epochs = []
    time_elapses = []
    epoch_indices = []

    # Continuous monitoring metrics
    continual_train_losses = []
    continual_train_indices = []
    test_indices = [epoch*len(train_data_loader.dataset) for epoch in range(number_of_epochs+1)]

    epoch_indices.append(0)
    # Test error without training the model # Epoch 0
    train_loss, train_accuracy = test_network(model=model,test_data_loader=train_data_loader)
    tr_losses_epochs.append(train_loss)
    tr_acc_epochs.append(train_accuracy)

    test_loss, test_accuracy = test_network(model=model,test_data_loader=test_data_loader)
    ts_losses_epochs.append(test_loss)
    ts_acc_epochs.append(test_accuracy)

    #Train the network for number of epochs
    for epoch in range(1, number_of_epochs+1, 1):
        epoch_indices.append(epoch)
        losses, counter = train_network_single_epoch(model=model,
                                                     train_data_loader=train_data_loader,
                                                     optimizer=optimizer,
                                                     log_interval = log_interval,
                                                     epoch = epoch, batch_size=train_batch_size)
        # Store this continous info
        continual_train_losses.extend(losses)
        continual_train_indices.extend(counter)

        # Store the losses and accuracy of test data on each epoch
        test_loss, test_accuracy = test_network(model=model,test_data_loader=test_data_loader)
        ts_losses_epochs.append(test_loss)
        ts_acc_epochs.append(test_accuracy)

        # Store the losses and accuracy of test data on each epoch
        train_loss, train_accuracy = test_network(model=model,test_data_loader=train_data_loader)
        tr_losses_epochs.append(train_loss)
        tr_acc_epochs.append(train_accuracy)
        
        # Store the time elapsed
        time_elapsed = timeit.timeit() - start
        time_elapses.append(time_elapsed)

    time_elapsed = timeit.timeit() - start
    return (epoch_indices, tr_losses_epochs, ts_losses_epochs, tr_acc_epochs, ts_acc_epochs,
            time_elapses, time_elapsed, continual_train_indices, continual_train_losses, test_indices)


def quick_train_and_analyse_model(model: nn.Module, train_data_loader: DataLoader,
                                  test_data_loader: DataLoader, optimizer: optim,
                                  log_interval: int = None,  number_of_epochs: int = 1,
                                  train_batch_size: int = 64):
    """This function trains the given model for specified epochs on trainset
    and provides the train_error, test_error, train_accuracy, test_accuracy, time_elpased"""
    start = time.time()

    #Train the network for number of epochs
    for epoch in range(1, number_of_epochs+1, 1):
        _, _ = train_network_single_epoch(model=model,
                                                     train_data_loader=train_data_loader,
                                                     optimizer=optimizer,
                                                     log_interval = log_interval,
                                                     epoch = epoch, batch_size=train_batch_size)

    # Evaluate and Store the losses and accuracy of test data
    test_loss, test_accuracy = test_network(model=model,test_data_loader=test_data_loader)

     # Evaluate and Store the losses and accuracy of test data on each epoch
    train_loss, train_accuracy = test_network(model=model,test_data_loader=train_data_loader)

    # Calculate the time elapsed
    time_elapsed = time.time() - start
    
    return (train_accuracy, train_loss, test_accuracy, test_loss, time_elapsed)
