import chess
import torch
import numpy as np
import pandas as pd

# own files
from chess_csv import WHITE_MOVES_CSV, BLACK_MOVES_CSV
from chess_annotation import bitboard_to_byteboard


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


def to_tensor(sample, grad=False):

    return torch.tensor(sample, requires_grad=grad)


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


def targets_to_numericals(color) -> list:
    """
    Decodes every move in a file by taking its index and returning it with a dict.
    :param color: which set of moves should be decoded
    """

    targets = []
    if type(color) is not bool:
        raise ValueError(f"Expected type {chess.COLORS} but received type {type(color)}")
    if color is True:
        targets = pd.read_csv(WHITE_MOVES_CSV, usecols=[0], header=0)
    if color is False:
        targets = pd.read_csv(BLACK_MOVES_CSV, usecols=[0], header=0)

    # getting the column values by taking the name of the first column as the key
    targets = targets[targets.columns[0]]
    targets.to_list()

    return targets


def tensor_to_targets(tensor: torch.Tensor, targets: list, amount_targets=1) -> list[str]:
    """
    Return the fitting tensor (for further calculations) or the fitting notation (for generating moves)
    to a given tensor based off a given dictionary.
    :param tensor: input tensor with non integer values
    :param targets: a dictionary created by targets_to_numericals containing a list of all UCI moves
    :param amount_targets: how big the amount of highest ranking annotations should be
    """

    # getting the highest index / indices of a given output tensor
    match = get_highest_indices(tensor)

    annotations = []
    # returns all fitting annotations to the tensors
    for i in range(amount_targets):
        annotations.append(targets[match[i]])

    return annotations


def get_highest_indices(iterable) -> [int]:
    """
    Returns the indices of the highest values of a given iterable.
    :param iterable: object which should be inspected (e.g. tensor, array, list)
    """
    return np.argsort(iterable).tolist()[::-1]


def create_targets_by_index(index, size):
    """
    Creates a target tensor by changing a given index of the possible outputs to 1.
    :param index: the index of the highest value
    :param size: size of the output tensor
    """
    # creating a tensor of all zeros with the size of the current outputs
    tensor = torch.zeros(size)
    # setting the target to 1 at index for current label
    tensor[index] = 1

    return tensor


def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """
    Compares if 2 tensors contain the same values.
    :param tensor1: first tensor
    :param tensor2: second tensor
    """
    for i, num in enumerate(tensor1):
        if num != tensor2[i]:
            return False

    return True
