import chess
import pandas as pd
import numpy as np

from chess_csv import white_games_csv, black_games_csv
import chess_annotation
from chess_annotation import bitboard_to_byteboard
import data_transformations as dt

from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, img_dir, color, transform=None, target_transform=None):
        # load the specified chess data from a csv file
        self.data, self.labels = prepare_chess_data(img_dir)
        self.color = color
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.labels)

    def __color__(self) -> chess.COLORS:
        return self.color

    def __getitem__(self, idx):
        label = self.labels[idx]
        byte_image = self.data[idx]
        # 12 channels
        bit_image = np.empty([12, 8, 8], np.float32)

        # transforming bitboards
        for i in range(12):
            temp_holder = bitboard_to_byteboard(byte_image[i])
            c = 0

            for j in range(64):
                if j % 8 == 0 and j != 0:
                    c += 1
                bit_image[i][c][j % 8] = temp_holder[j]

        # turning the np.array into a pytorch tensor
        if self.transform:
            bit_image = self.transform(bit_image)

        # bit_image = bit_image.permute(0, 4, 1, 2, 3)
        # return the bitboards and the label as a tensor
        return bit_image, dt.string_to_tensor(label)


class ChessTestData(Dataset):
    def __init__(self, game_state: np.array, transform=None):
        self.data = [game_state]
        self.labels = [0]
        self.transform = transform

    def __len__(self) -> int:
        return 1  # Batch size will always be 1 as we are only loading 1 game state at a time

    def __getitem__(self, idx):
        label = self.labels[idx]
        byte_image = self.data[idx]
        # 12 channels
        bit_image = np.empty([12, 8, 8], np.float32)

        # transforming bitboards
        for i in range(12):
            temp_holder = bitboard_to_byteboard(byte_image[i])
            c = 0

            for j in range(64):
                if j % 8 == 0 and j != 0:
                    c += 1
                bit_image[i][c][j % 8] = temp_holder[j]

        # turning the np.array into a pytorch tensor
        if self.transform:
            bit_image = self.transform(bit_image)

        # bit_image = bit_image.permute(0, 4, 1, 2, 3)
        # return the bitboards and the label as a tensor
        return bit_image, label


def prepare_chess_data(path: str):
    """
    This function will convert a csv file into usable data
    :param path: path to the CSV file
    """

    # change the base path based on where the CSV folder is located
    base_path = "/path/to/folder/"

    bitboards = np.array(pd.read_csv(base_path + path, usecols=range(12)))

    labels = np.array(pd.read_csv(base_path + path, usecols=[12]))

    return bitboards, labels


def init_chess_dataset(color: chess.COLORS) -> ChessDataset:
    if color is not True and color is not False:
        raise ValueError(f"Variable color must be of type {chess.COLORS} but is of type {type(color)}!")

    if color:
        dataset = ChessDataset(black_games_csv,
                               color,
                               transform=dt.to_tensor)  # dt.RandomCrop(4) additionally
        return dataset

    elif not color:
        dataset = ChessDataset(white_games_csv,
                               color,
                               transform=dt.to_tensor)
        return dataset


def init_chess_testset(fen):
    bit_image = chess_annotation.fen_to_bitboards(fen)

    dataset = ChessTestData(bit_image)
    return dataset
