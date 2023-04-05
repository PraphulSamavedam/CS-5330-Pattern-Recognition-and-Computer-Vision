"""
Written by: Samavedam Manikhanta Praphul
Version: 1.0
Description: 
This file requests for the user input for file selection and provides prediction 
based on the model trained. 
"""

from test import load_model
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from PIL import Image
from torchvision import transforms
import torch
from torch.nn import Module


def predict(model: Module, file_path: str):
    """This function predicts the digit in the file based on the model.
    Params
    model: Module trained neural network module
    file_path: str path of the image file on which prediction needs to be performed.
    """
    image = Image.open(file_path, mode="r")
    resized_image = image.resize((28, 28))  # Resizing to fit the shape of dimensions.
    resized_image = resized_image.convert(
        mode="L"
    )  # Converting to black and white image.
    tensor = transforms.PILToTensor()(resized_image)  # Convert to Tensor
    tensor = tensor.to(dtype=torch.float)
    tensor.unsqueeze_(dim=0)  # To makeup for the batch sample
    output = model(tensor)
    prediction = output.data.max(1, keepdim=True)[1]
    return prediction.item()


def main():
    """This is the function which runs when run as a standalone script.
    Returns
    0 if the script exits successfully."""
    model = load_model("models/final_model.pth")
    # Get the data from the user
    Tk().withdraw()
    multiple_file_names = askopenfilenames()
    predictions = []
    correct_predictions = 0
    for file_name in multiple_file_names:
        print(f"For file :{file_name}")
        prediction = predict(model=model, file_path=file_name)
        ground_truth = file_name.split("/")[-1].split("_")[0]
        correct_predictions += str(ground_truth) == str(prediction)
        predictions.append(prediction)
        print(f"Ground Truth:{ground_truth} Prediction:{prediction}")
        print(f"Correct predictions till now: {correct_predictions}")
    print(
        f"Accuracy on custom images:{100*correct_predictions/len(multiple_file_names):.2f}%"
    )
    return 0


if __name__ == "__main__":
    main()
