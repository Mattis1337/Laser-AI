import chess as c
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

# Importing own files
from chess_csv import white_games_csv, black_games_csv, white_moves_csv, black_moves_csv, local_csv_path
import chess_annotation
from chess_annotation import bitboard_to_byteboard
import data_transformations as dt


class ChessDataset(Dataset):
    def __init__(self, img_dir, color, transform=None):
        # load the specified chess data from a csv file
        self.data, self.labels = prepare_chess_data(img_dir)
        self.color = color
        self.transform = transform
        self.targets_transformed = dt.targets_to_numericals(color)

    def __len__(self) -> int:
        return len(self.labels)

    def __color__(self) -> c.COLORS:
        return self.color

    def __gettargets__(self) -> dict:
        return self.targets_transformed

    def __getitem__(self, idx):
        label = self.labels[idx]
        label = label[0]
        bitboards = self.data[idx]

        byteboards = dt.transform_bitboards(bitboards)

        # turning the np.array into a pytorch tensor
        if self.transform:
            byteboards = self.transform(byteboards)

        # getting the transformed target of the label
        if label in self.targets_transformed:
            target = self.targets_transformed[label]
        else:
            raise ValueError(f"Target for label not found in targets_transformed: {label} (label)!",
                             "Update file containing all moves!")

        # return the bitboards and the label as a tensor
        return byteboards, target


def prepare_chess_data(path: str):
    """
    This function will convert a csv file into a np.array()
    :param path: path to the CSV file
    """
    # change the base path based on where the CSV folder is located
    base_path = local_csv_path

    bitboards = np.array(pd.read_csv(base_path + path, usecols=range(12)))

    labels = np.array(pd.read_csv(base_path + path, usecols=[12]))  # new method turns label into tensor

    return bitboards, labels


def init_chess_dataset(color: c.COLORS) -> ChessDataset:
    if color is not True and color is not False:
        raise ValueError(f"Variable color must be of type {c.COLORS} but is of type {type(color)}!")

    if color is True:
        dataset = ChessDataset(white_games_csv,
                               color,
                               transform=dt.to_tensor)  # dt.RandomCrop(4) additionally
        return dataset

    elif color is False:
        dataset = ChessDataset(black_games_csv,
                               color,
                               transform=dt.to_tensor)
        return dataset


def get_output_length(color: c.COLORS) -> int:
    """Getting the total number of learnable moves"""
    base_path = "/home/mattis/development/"
    df = None

    if type(color) is not bool:
        raise ValueError(f"Expected type {c.COLORS} but received type {type(color)}")

    if color is True:
        df = pd.read_csv(base_path + white_moves_csv)
    if color is False:
        df = pd.read_csv(base_path + black_moves_csv)

    return df.__len__()
