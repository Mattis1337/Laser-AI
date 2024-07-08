import chess
import pandas as pd
import numpy as np

from chess_csv import white_games_csv, black_games_csv
from chess_annotation import bitboard_to_byteboard
import data_transformations as dt

from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, img_dir, batch_size, iter_num, color, transform=None, target_transform=None):
        # Get the file containing all white moves
        self.data, self.labels = prepare_chess_data(img_dir, batch_size, iter_num)
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
        # 3 channels
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


def prepare_chess_data(path: str, batch: int, iteration: int):
    """
    This function will convert a csv file into usable data
    :param path: path to the CSV file
    :param batch: the number of games to be loaded
    :param iteration: the number of batches/iterations which have already been conducted over the dataframe
    """

    df = pd.read_csv("/home/mattis/development/" + path)  # TODO: change this line depending on your own path

    # create arrays with the according sizes  including the absolute size of the dataframe
    bitboards = np.empty((batch, 12))
    labels = []

    for i in range(batch):
        if df.values[i] is None: break

        bitboards[i][0] = df.values[i + (iteration * batch)][0]
        bitboards[i][1] = df.values[i + (iteration * batch)][1]
        bitboards[i][2] = df.values[i + (iteration * batch)][2]
        bitboards[i][3] = df.values[i + (iteration * batch)][3]
        bitboards[i][4] = df.values[i + (iteration * batch)][4]
        bitboards[i][5] = df.values[i + (iteration * batch)][5]
        bitboards[i][6] = df.values[i + (iteration * batch)][6]
        bitboards[i][7] = df.values[i + (iteration * batch)][7]
        bitboards[i][8] = df.values[i + (iteration * batch)][8]
        bitboards[i][9] = df.values[i + (iteration * batch)][9]
        bitboards[i][10] = df.values[i + (iteration * batch)][10]
        bitboards[i][11] = df.values[i + (iteration * batch)][11]

        labels.append(df.values[i + (iteration * batch)][12])

    return bitboards, labels


def init_chess_dataset(color: chess.COLORS, batch_size: int,  iter_num: int) -> ChessDataset:
    if color is not True and color is not False:
        raise ValueError(f"Variable color must be of type {chess.COLORS} but is of type {type(color)}!")

    if color:
        dataset = ChessDataset(black_games_csv,
                               batch_size,
                               iter_num,
                               color,
                               transform=dt.ToTensor())  # dt.RandomCrop(4) additionally
        return dataset

    elif not color:
        dataset = ChessDataset(white_games_csv,
                               batch_size,
                               iter_num, color,
                               transform=dt.ToTensor())
        return dataset
