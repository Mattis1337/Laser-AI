import chess
import pandas as pd
import numpy as np

from chess_csv import white_games_csv, black_games_csv
from chess_annotation import bitboard_to_byteboard

from torch.utils.data import Dataset


# TODO: Adjust the custom dataset to actually taking bitboards as input and not pictures:
#  -look at what output the transform type ToTensor offers and adjust the bitboards accordingly
#  -also look at vgg and what that does
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

    # TODO: turn bitboards into byteboards before parsing them to the network via this function
    def __getitem__(self, idx: int):
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        label = self.labels[idx]
        byte_image = self.data[idx]
        bit_image = np.empty([12, 64])  # read_image converts into rgb or grayscale, so maybe load nums accordingly

        for i in range(12):
            temp_holder = bitboard_to_byteboard(byte_image[i])

            for j in range(64):
                bit_image[i][j] = temp_holder[j]
        # if self.transform:
            # image = self.transform(image)  # idk what the fuck this is
        # if self.target_transform:
            # label = self.target_transform(label)
        return bit_image, label


def prepare_chess_data(path: str, batch: int, iter: int):
    """
    This function will convert a csv file into usable data
    :param path: path to the CSV file
    :param batch: the number of games to be loaded
    :param iter: the number of batches/iterations which have already been conducted over the dataframe
    """
    df = pd.read_csv("/home/user/path/to/" + path)  # TODO: change this line depending on your own path

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


def init_chess_dataset(color: chess.COLORS,batch_size: int,  iter_num: int) -> ChessDataset:
    if color is not True and color is not False:
        raise ValueError(f"Variable color must be of type {chess.COLORS} but is of type {type(color)}!")

    if color:
        dataset = ChessDataset(black_games_csv, batch_size, iter_num, color)
        return dataset
    elif not color:
        dataset = ChessDataset(white_games_csv, batch_size, iter_num, color)
        return dataset
