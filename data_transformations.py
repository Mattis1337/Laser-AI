import torch
import numpy as np


#  https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html

# TODO: Adjust class to custom Dataset
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample[0], sample[1]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose()
        return torch.from_numpy(image)


def string_to_tensor(str_list):
    # setting the input dimensions to a max of 8 chars
    input_dimension = [8, 1]
    # parsing the only item in the list
    string = str_list[0]
    # initialising the Float Tensor
    proc_tensor = torch.zeros(input_dimension, dtype=torch.float)

    for i in range(len(string)):
        char = string[i]
        # getting the unicode of a char
        char_as_bytes = ord(char)
        # adding the encoded char to the tensor as a float
        proc_tensor[i] = float(char_as_bytes)

    return proc_tensor
