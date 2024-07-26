import chess
import torch
import numpy as np
import pandas as pd

# own files
from chess_csv import black_moves_csv, white_moves_csv, local_csv_path
from chess_annotation import bitboard_to_byteboard
import datasets


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


def to_tensor(sample):
    return torch.from_numpy(sample)


def transform_bitboards(bitboards):
    # 12 channels
    byteboard = np.empty([12, 8, 8], np.float32)

    # transforming bitboards
    for i in range(12):
        temp_holder = bitboard_to_byteboard(bitboards[i])
        c = 0

        for j in range(64):
            if j % 8 == 0 and j != 0:
                c += 1
            byteboard[i][c][j % 8] = temp_holder[j]
    return byteboard


# TODO: new function which decodes the targets
def targets_to_tensor(color) -> dict:
    # setting the input dimensions to a max of 8 chars
    input_dimension = [datasets.get_output_length(color)]
    targets = []
    if type(color) is not bool:
        raise ValueError(f"Expected type {chess.COLORS} but received type {type(color)}")
    if color is True:
        targets = pd.read_csv(local_csv_path + white_moves_csv, usecols=[0])
    if color is False:
        targets = pd.read_csv(local_csv_path + black_moves_csv, usecols=[0])

    target_tensor_dict = {}

    targets = targets.to_numpy()

    for i, target in enumerate(targets):
        target = target[0]
        # initialise the output for the network using zeros
        tensor = torch.zeros(input_dimension)
        # setting the target to 1 at index for current label
        tensor[i] = 1
        # changing tensor to torch.float32
        tensor = tensor.to(torch.float32)
        # adding the target tensor with the key of the fitting move
        target_tensor_dict[target] = tensor

    return target_tensor_dict


def tensor_to_targets(tensor: torch.Tensor, targets: dict, annotation=False, amount_targets=1):
    """
    Return the fitting tensor (for further calculations) or the fitting notation (for generating moves)
    to a given tensor based off a given dictionary.
    :param tensor: input tensor with non integer values
    :param targets: a dictionary created by targets_to_tensor containing fitting notation to a possible target tensor
    :param annotation: whether the output should be the notation or the tensor (by default the tensor)
    :param amount_targets: how big the amount of highest ranking annotations should be
    """
    ...
