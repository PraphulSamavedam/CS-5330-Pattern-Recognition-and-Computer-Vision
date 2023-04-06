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