import chess
import pandas as pd
import numpy as np

from chess_csv import white_games_csv, white_moves_csv, black_games_csv, black_moves_csv
from chess_annotation import bitboard_to_byteboard

from torch.utils.data import Dataset


# TODO: Adjust the custom dataset to actually taking bitboards as input and not pictures:
#  -look at what output the transform type ToTensor offers and adjust the bitboards accordingly
#  -also look at vgg and what that does
class ChessDataset(Dataset):
    def __init__(self, img_labels, img_dir, color, transform=None, target_transform=None):
        # Get the file containing all white moves
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.color = color
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.img_labels)

    def __color__(self) -> chess.COLORS:
        return self.color

    # TODO: turn bitboards into byteboards before parsing them to the network via this function
    def __getitem__(self, idx) -> tuple[list, str]:
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        label = self.img_labels.iloc[idx, 1]
        image = None  # read_image converts into rgb or grayscale, so maybe load nums accordingly
        if self.transform:
            image = self.transform(image)  # idk what the fuck this is
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def prepare_chess_data(path: str, batch: int, iter: int):
    """
    This function will convert a csv file into usable data
    :param path: path to the CSV fil
    :param batch: the number of games to be loaded
    :param iter: the number of batches/iterations which have already been conducted over the dataframe
    """
    df = pd.read_csv(path)

    # create arrays with the according sizes  including the absolute size of the dataframe
    bitboards = np.empty([batch, 12], np.int64)
    labels = []

    for i in range(batch):
        if df.values[i] is None: break

        bitboards[i][0] = df.values[i + (iter * batch)][0]
        bitboards[i][1] = df.values[i + (iter * batch)][1]
        bitboards[i][2] = df.values[i + (iter * batch)][2]
        bitboards[i][3] = df.values[i + (iter * batch)][3]
        bitboards[i][4] = df.values[i + (iter * batch)][4]
        bitboards[i][5] = df.values[i + (iter * batch)][5]
        bitboards[i][6] = df.values[i + (iter * batch)][6]
        bitboards[i][7] = df.values[i + (iter * batch)][7]
        bitboards[i][8] = df.values[i + (iter * batch)][8]
        bitboards[i][9] = df.values[i + (iter * batch)][9]
        bitboards[i][10] = df.values[i + (iter * batch)][10]
        bitboards[i][11] = df.values[i + (iter * batch)][11]

        labels.append(df.values[i + (iter * batch)][12])

    return bitboards, labels


def init_chess_dataset(color: chess.COLORS) -> ChessDataset:
    if not isinstance(color, chess.COLORS):
        raise ValueError(f"Variable color must be of type {chess.COLORS} but is of type {type(color)}!")
    if color:
        dataset = ChessDataset(black_moves_csv, black_games_csv, color)
        return dataset
    elif not color:
        dataset = ChessDataset(white_moves_csv, white_games_csv, color)
        return dataset
