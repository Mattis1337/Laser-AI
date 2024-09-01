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
    # setting the input dimensions
    output_dimension = [datasets.get_output_length(color)]
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
        tensor = create_targets_by_index(i, output_dimension)
        # adding the target tensor with the key of the fitting move
        target_tensor_dict[target] = tensor

    return target_tensor_dict


def tensor_to_targets(tensor: torch.Tensor, color: chess.COLORS, targets: dict, annotation=False, amount_targets=1):
    """
    Return the fitting tensor (for further calculations) or the fitting notation (for generating moves)
    to a given tensor based off a given dictionary.
    :param tensor: input tensor with non integer values
    :param color: color of neural network to get the size of the outputs
    :param targets: a dictionary created by targets_to_tensor containing fitting notation to a possible target tensor
    :param annotation: whether the output should be the notation or the tensor (by default the tensor)
    :param amount_targets: how big the amount of highest ranking annotations should be
    """

    match = []
    output_dimension = [datasets.get_output_length(color)]

    # getting the highest index / indices of a given output tensor
    for i in range(amount_targets):
        if i == 0:
            match.append(get_highest_index(tensor[0]))
        else:
            match.append(get_highest_index(tensor[0], match))

    tensors = []
    # creating the tensors based on which index should be the right output
    for i in range(amount_targets):
        tensors.append(create_targets_by_index(match[i], output_dimension))

    # checks if tensors should be turned into fitting annotation
    if annotation is False:
        return torch.Tensor(tensors[0]).float()  # returning only index 0

    annotations = []
    # returns all fitting annotations to the tensors
    for i in range(amount_targets):
        for key, value in targets.items():
            if compare_tensors(value, tensors[i]):
                annotations.append(key)

    return annotations


def get_highest_index(iterable, skips=None):
    """
    Returns the index of the highest value of a given iterable.
    :param iterable: object which should be used for iterating (e.g. tensor, array, list)
    :param skips: indices which have already been evaluated and should be skipped
    """
    highest = 0.0
    match = None

    if skips is None:
        # setting highest to the highest number found in the iterable
        highest = max(iterable)
        for i in range(iterable.__len__()):
            # looking for highest number will mitigate time consumption
            if iterable[i] == highest:
                return i

        return match

    if skips:
        for i in range(iterable.__len__()):
            if any(index == i for index in skips): continue

            if iterable[i] >= highest:
                highest = iterable[i]
                match = i

        return match


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
    for i, index in enumerate(tensor1):
        if index != tensor2[i]:
            return False

    return True
