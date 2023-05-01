"""Written by: Samavedam Manikhanta Praphul
Version: 1.0
Description:
This file has all the data transform classes required for the project. 
"""
import torchvision

class GreekTransform:
    """This class represents the transformation required to convert 133 x 133 color images of
     greek letters to 28 x28 grayscale images.
     """
    def __init__(self) -> None:
        pass

    def __call__(self, x_input) :
        x_input = torchvision.transforms.functional.rgb_to_grayscale(x_input)
        x_input = torchvision.transforms.functional.affine(x_input, 0, (0, 0), 36/128, 0)
        x_input = torchvision.transforms.functional.center_crop(x_input, (28, 28))
        return torchvision.transforms.functional.invert(x_input)

