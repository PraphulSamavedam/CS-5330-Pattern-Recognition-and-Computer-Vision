"""
Written by: Samavedam Manikhanta Praphul
Version: 1.0
Description: 
This file has definitions of neural networks which are written from scratch in this project. 
"""
# class definitions
from torch import nn


class BaseNetwork(nn.Module):
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
        super(BaseNetwork,self).__init__()
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
    

class NetWorkKernel1(nn.Module):
    '''
    This neural network model is similar to BaseNetwork except that kernels are of size 1 instead of 5
    '''

    def __init__(self):
        """This initializes my network with the required parameters which needs to be learned."""
        # Convolution stack
        super(NetWorkKernel1, self).__init__()
        self.convolution_stack = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(1,1)) # Output shape ( 10, 28, 28)
                                               ,nn.MaxPool2d(kernel_size=2, stride = 2, padding=0) # Output shape (10, 14, 14)
                                               ,nn.ReLU()
                                               ,nn.Conv2d(in_channels=10, out_channels=20, kernel_size = 1) # Output shape (20, 14, 14)
                                               ,nn.Dropout2d(p=0.5)
                                               ,nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # Output shape (20, 7, 7 )
                                               ,nn.ReLU()
                                               ,nn.Flatten(start_dim=1, end_dim=-1) # Output shape 7*7*20 = 980
                                              )
        # Classification stack
        self.classification_stack = nn.Sequential(nn.Linear(in_features=980, out_features=50)
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
    

class ManyParallelFiltersNetWork(nn.Module):
    '''
    This neural network model is similar to BaseNetwork except that number of 
    filters in the convolution layer are many hence name many parallel filters network.
    The number of parallel filters is based on the parameter passed during instantiation
    '''

    def __init__(self, conv_channels):
        """This initializes my network with the required parameters which needs to be learned."""
        super(ManyParallelFiltersNetWork, self).__init__()

        ### Convolution stack
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=5)
        # Output shape ( conv_channels, 24, 24)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        # Output shape ( conv_channels, 12, 12)
        self.conv2 = nn.Conv2d(in_channels=conv_channels, out_channels=2*conv_channels,
                               kernel_size = 5)
        # Output shape (2*conv_channels, 8, 8)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        # Output shape (2*conv_channels, 4, 4 )
        self.flat = nn.Flatten(start_dim=1, end_dim=-1)
        # Output shape = 4* 4* 2* conv_channels


        # Classification stack
        self.fc1 = nn.Linear(in_features= 32*conv_channels, out_features=50)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=50, out_features=10)
        self.classify = nn.LogSoftmax(dim=None)

    # computes a forward pass for the network
    def forward(self, data):
        """ This function computes a forward pass through the network."""

        # Pass through convolution stack
        data = self.conv1(data)
        data = self.max_pool1(data)
        data = self.relu1(data)
        data = self.conv2(data)
        data = self.dropout1(data)
        data = self.max_pool2(data)
        data = self.relu2(data)
        data = self.flat(data)

        # Pass through classification stack
        data = self.fc1(data)
        data = self.relu3(data)
        data = self.fc2(data)
        return self.classify(data)
    
class DeepNetwork1(nn.Module):
    '''
    This neural network model is similar to BaseNetwork except that each of the convolution layer
    There is additional layer which process the output of the convolution again, maintaining 
    the expected shape through proper padding. 
    '''

    def __init__(self):
        """This initializes my network with the required parameters which needs to be learned."""
        super(DeepNetwork1, self).__init__()

        ### Convolution stack
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, padding=0)
        self.addl1= nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, padding=2)
        # Output shape ( 10, 24, 24)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        # Output shape ( 10, 12, 12)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size = 5)
        self.addl2= nn.Conv2d(in_channels=20, out_channels=20, kernel_size=5, padding=2)
        # Output shape (20, 8, 8)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        # Output shape (20, 4, 4 )
        self.flat = nn.Flatten(start_dim=1, end_dim=-1)
        # Output shape = 4*4*20

        # Classification stack
        self.fc1 = nn.Linear(in_features= 320, out_features=50)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=50, out_features=10)
        self.classify = nn.LogSoftmax(dim=None)

    # computes a forward pass for the network
    def forward(self, data):
        """ This function computes a forward pass through the network."""

        # Pass through convolution stack
        data = self.conv1(data)
        data = self.addl1(data)
        data = self.max_pool1(data)
        data = self.relu1(data)
        data = self.conv2(data)
        data = self.addl2(data)
        data = self.dropout1(data)
        data = self.max_pool2(data)
        data = self.relu2(data)
        data = self.flat(data)

        # Pass through classification stack
        data = self.fc1(data)
        data = self.relu3(data)
        data = self.fc2(data)
        return self.classify(data)

class DeepNetwork2(nn.Module):
    '''
    This neural network model is similar to BaseNetwork except that each of the convolution layer
    There are 2 additional layers which process the output of the convolution again, maintaining 
    the expected shape through proper padding. 
    '''

    def __init__(self):
        """This initializes my network with the required parameters which needs to be learned."""
        super(DeepNetwork2, self).__init__()

        ### Convolution stack
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, padding=0)
        self.addl1= nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, padding=2)
        self.addl2= nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, padding=2)
        # Output shape ( 10, 24, 24)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        # Output shape ( 10, 12, 12)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size = 5)
        self.addl3= nn.Conv2d(in_channels=20, out_channels=20, kernel_size=5, padding=2)
        self.addl4= nn.Conv2d(in_channels=20, out_channels=20, kernel_size=5, padding=2)
        # Output shape (20, 8, 8)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        # Output shape (20, 4, 4 )
        self.flat = nn.Flatten(start_dim=1, end_dim=-1)
        # Output shape = 4*4*20

        # Classification stack
        self.fc1 = nn.Linear(in_features= 320, out_features=50)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=50, out_features=10)
        self.classify = nn.LogSoftmax(dim=None)

    # computes a forward pass for the network
    def forward(self, data):
        """ This function computes a forward pass through the network."""

        # Pass through convolution stack
        data = self.conv1(data)
        data = self.addl1(data)
        data = self.addl2(data)
        data = self.max_pool1(data)
        data = self.relu1(data)
        data = self.conv2(data)
        data = self.addl3(data)
        data = self.addl4(data)
        data = self.dropout1(data)
        data = self.max_pool2(data)
        data = self.relu2(data)
        data = self.flat(data)

        # Pass through classification stack
        data = self.fc1(data)
        data = self.relu3(data)
        data = self.fc2(data)
        return self.classify(data)
