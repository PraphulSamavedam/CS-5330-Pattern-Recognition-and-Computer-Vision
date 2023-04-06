"""
Written by: Samavedam Manikhanta Praphul
Version: 1.0
Description: 
This file requests for file requests user for file selection & provides prediction 
based on the trained model selected. 
"""

import torchvision
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import freeze_layers_and_modify_last_layer
from transfer_greek import GreekTransform
from models import BaseNetwork
from test_basic import load_model


def test_transfer_learning(model:Module, test_data_loader:DataLoader):
    """This function calculates the performance on test data set provided
    and displays the prediction
    Returns accuracy of model
    """
    model.eval() # To set the model in evaluation mode -> No training
    test_loss:float = 0
    correct_predictions:int = 0
    mapping = {0:"Alpha", 1:"Beta", 2:"Gamma"}
    rows = 3
    cols = 3
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.suptitle("Predictions on custom data of greek letters")
    with torch.no_grad():
        index = 1
        for data, target in test_data_loader:
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output, target, reduction="sum").item()
            prediction = output.data.max(1, keepdim=True)[1]
            correct_predictions += prediction.eq(target.data.view_as(prediction)).sum()
            for i, tgt in enumerate(target):
                plt.subplot(rows, cols, index)
                plt.axis("off")
                plt.xticks=[]
                plt.yticks=[]
                plt.imshow(data[i].squeeze(), cmap='gray')
                plt.title(f"Truth:{mapping[tgt.item()]}, Prediction:{mapping[prediction[i].item()]}")
                index +=1
        # Average the loss over all the samples
        test_loss /= len(test_data_loader.dataset)
        accuracy = 100. * correct_predictions / len(test_data_loader.dataset)
        print(f"""\nTest set: Avg. loss: {test_loss:.4f},
                    Accuracy: {correct_predictions}/{len(test_data_loader.dataset)} ({accuracy:.2f}%)\n""")
    plt.show()
    return test_loss, accuracy


def main():
    """This is the function which runs when run as a standalone script.
    Returns
    0 if the script exits successfully."""

    # Start with base model
    model = load_model(BaseNetwork, "models/final_model.pth")
    # Modify the model to freeze all layers & modify last FC layer to fit
    model = freeze_layers_and_modify_last_layer(model=model, output_features=3)

    # Load the Transfer Learned network with learned weights
    model_state_dict = torch.load("models/model_greek.pth")
    model.load_state_dict(model_state_dict)

    test_data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(root="data/greek_test",
                                         transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            GreekTransform(),
                                            torchvision.transforms.Normalize((0.1307,),
                                                                             (0.3801,),)
                                         ])), batch_size = 3, shuffle=True )
       
    _, accuracy = test_transfer_learning(model=model, test_data_loader=test_data_loader)
    print(f"Accuracy on custom images:{accuracy:.2f}%")
    return 0


if __name__ == "__main__":
    main()
