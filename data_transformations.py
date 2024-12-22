import chess
import torch
import numpy as np
import pandas as pd

# own files
from chess_csv import BLACK_MOVES_CSV, BLACK_GAMES_CSV, WHITE_MOVES_CSV, WHITE_GAMES_CSV
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


def targets_to_numericals(color) -> dict:
    """
    Decodes every move in a file by taking its index and returning it with a dict.
    :param color: which set of moves should be decoded
    """

    targets = []
    if type(color) is not bool:
        raise ValueError(f"Expected type {chess.COLORS} but received type {type(color)}")
    if color is True:
        targets = pd.read_csv(WHITE_MOVES_CSV, usecols=[0])
    if color is False:
        targets = pd.read_csv(BLACK_MOVES_CSV, usecols=[0])

    target_tensor_dict = {}

    targets = targets.to_numpy()

    for i, target in enumerate(targets):
        target = target[0]
        # adding the target tensor with the key of the fitting move
        target_tensor_dict[target] = i

    return target_tensor_dict


def tensor_to_targets(tensor: torch.Tensor, targets: dict, amount_targets=1):
    """
    Return the fitting tensor (for further calculations) or the fitting notation (for generating moves)
    to a given tensor based off a given dictionary.
    :param tensor: input tensor with non integer values
    :param targets: a dictionary created by targets_to_numericals containing fitting notation to a possible target tensor
    :param amount_targets: how big the amount of highest ranking annotations should be
    """

    # getting the highest index / indices of a given output tensor
    match = get_highest_index(tensor[0], amount_targets)[-amount_targets:]
    # reversing the list so the highest index is now at [0]
    match = match[::-1]

    annotations = []
    # returns all fitting annotations to the tensors
    for i in range(amount_targets):
        for key, value in targets.items():
            if value == match[i]:
                annotations.append(key)

    return annotations


def get_highest_index(iterable, amount_targets: int) -> list[int]:
    """
    Returns the index of the highest value of a given iterable.
    :param iterable: object which should be inspected (e.g. tensor, array, list)
    :param amount_targets: how long the list of highest indices should be
    """
    sorted_idx = []
    for i, val in enumerate(iterable):
        if len(sorted_idx) < amount_targets:
            sorted_idx.append(i)
        for j in range(len(sorted_idx)):
            if iterable[sorted_idx[j]] <= val:
                sorted_idx.insert(j, i)
                break
        if len(sorted_idx) > amount_targets:
            sorted_idx.pop(-1)
    return sorted_idx


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
    # changing tensor to torch.float32
    tensor = tensor.to(torch.float32)

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
