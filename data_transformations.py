import torch
import numpy as np


# TODO: adjusting the functions to our desires
#  https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html

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
    # https://stackoverflow.com/questions/44617871/how-to-convert-a-list-of-strings-into-a-tensor-in-pytorch

    max_l = 0
    tensor_list = []  # list of tensors

    # turn str to byte and get the max byte size
    for sample in str_list:
        tensor_list.append(torch.ByteTensor(list(bytes(sample, 'utf8'))))
        max_l = max(tensor_list[-1].size()[0], max_l)

    # max_l will always be 1 and since the number of chars in a move notation can be up to 7 we take 8 for smoother
    # calculations
    input_dimension = [8, max_l]
    # turn the tensors into 1 tensor of uint8
    proc_tensor = torch.zeros(input_dimension, dtype=torch.float)  # processed tensor
    for i, tensor in enumerate(tensor_list):
        proc_tensor[i, 0:tensor.size()[0]] = tensor

    return proc_tensor
