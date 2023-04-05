"""
Written by: Samavedam Manikhanta Prahul
Version: 1.0
Description: This file has the utility functions that can be reused in other scripts. 
"""
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torchvision import transforms

def visualize_data_loader_data(data_loader: DataLoader, samples:int = 6, title:str=None):
    """This function visualizes the first n samples of the dataloader provided.
    Params
    data_loader: DataLoader data which needs to be visualized.
    samples:int = 6 number of samples of the data that you want to visualize.
    title:str = None, if title is None the data is presented with title 
    'Visualizing first #samples data points'
    Raises
    Assertion Error if samples are not in range of [1, 10]
    """
    assert(samples >= 1) & (samples<=10)
    fig = plt.figure(figsize=(10, 10))
    
    # Set the subplots configuration
    rows = 2
    cols = samples//rows
    if (rows* cols < samples):
        cols += 1
    
    title = f"Visualizing first {samples} data points" if title == "" else title
    # Get the data to plot
    enumerator = enumerate(data_loader)
    _, (img_data, labels) = next(enumerator)
    plt.title(title)
    plt.axis("off")

    # Plotting the data
    for index in range(1, samples+1):
        img = img_data[index]
        label = labels[index]
        fig.add_subplot(rows, cols, index)
        plt.title(f"Ground truth {label.item()}")
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")

    #Display the data
    plt.show()

def get_mnist_data_loaders(train_batch_size:int =64, test_batch_size:int=1000):
    """This function returns the data loaders corresponding to train_data and test_data"""
    train_data = MNIST(root='data/', download=True, train=True, transform = transforms.ToTensor())
    mean_val = (train_data.train_data.detach().float().mean()/255).item()
    stdv_val = (train_data.train_data.detach().float().std()/255).item()
    training_data_loader = DataLoader(MNIST(root="data/", download=True,train= True,
                                     transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((mean_val,), (stdv_val,))
                                     ])), batch_size = train_batch_size)

    testing_data_loader = DataLoader(MNIST(root="data/", download=True, train= False, 
                                     transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((mean_val, ), (stdv_val,))
                                     ])), shuffle = False # Shuffle turned off to ensure reproducibility
                                     ,batch_size=test_batch_size)
    
    return training_data_loader, testing_data_loader
