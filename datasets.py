import ast
import random
import os

import chess as c
import pandas as pd
import numpy as np
import torch
from pydantic.experimental.pipeline import transform

from torch.utils.data import Dataset

# Importing own files
from chess_csv import WHITE_GAMES_CSV, BLACK_GAMES_CSV, WHITE_MOVES_CSV, BLACK_MOVES_CSV
import data_transformations as dt


class ChessDataset(Dataset):
    def __init__(self, img_dir, color, rnn, transform=None, data=None, labels=None):
        # load the specified chess data from a csv fill
        # check whether chunking the data is already prepared as in the case of chunking
        try:
            if data or labels is None:
                self.data, self.labels = prepare_chess_data(img_dir, rnn)
        except ValueError:
            # if we get a ValueError params data and label contain a df whose real value is ambiguous
            self.data = data.to_numpy()
            self.labels = labels.to_numpy()
        
        self.rnn = rnn
        self.color = color
        self.transform = transform
        self.targets_transformed = {}
        targets = dt.targets_to_numericals(color)
        for i in range(len(targets)):
            self.targets_transformed[targets[i]] = i

        self.transformed_labels = []
        self.transformed_games = []

        if rnn is False:
            return

        self.__sample__()

    def __len__(self) -> int:
        if self.rnn is True:
            return len(self.transformed_labels)
        return len(self.labels)

    def __color__(self) -> c.COLORS:
        return self.color

    def __gettargets__(self) -> dict:
        return self.targets_transformed

    def __getitem__(self, idx):
        if self.rnn is True:
            return self.transformed_games[idx[0]], self.transformed_labels[idx[0]]

        label = self.labels[idx]
        label = label[0]
        bitboards = self.data[idx]

        byteboards = dt.transform_bitboards(bitboards)

        # turning the np.array into a pytorch tensor
        if self.transform:
            byteboards = self.transform(byteboards)

        # getting the transformed target of the label
        if label in self.targets_transformed:
            target_idx = self.targets_transformed[label]
            target = torch.zeros([len(self.targets_transformed)])
            target[target_idx] = 1
        else:
            raise ValueError(f"Target for label not found in targets_transformed: {label} (label)!",
                             "Update file containing all moves!")

        # return the bitboards and the label as a tensor
        return byteboards, target

    def __sample__(self, section_len=10_000):
        # sampling already loaded data
        sample = random.sample(range(len(self.data)), section_len)
        s_data = [self.data[x] for x in sample]
        s_labels = [self.labels[x] for x in sample]
        self.transformed_games = []
        self.transformed_labels = []

        # TODO: all of this is way too slow
        out_length = get_output_length(self.color)
        for idx in range(len(s_data)):
            bitboards = s_data[idx]
            label = s_labels[idx]
            label = label[0]
            bitboards = bitboards[0]
            byteboards = []
            for game in bitboards:
                byteboards.append(dt.transform_bitboards(game))
            # turning the list into a np.array so the transform works
            byteboards = np.array(byteboards)

            # if dataset should return a sequence of moves get each one from the hot encoded dict
            targets = np.empty([len(label), out_length])
            for i, l in enumerate(label):
                # getting the transformed target of the label
                if l in self.targets_transformed:
                    targets[i] = dt.create_targets_by_index(self.targets_transformed[l], out_length)
                else:
                    raise ValueError(f"Target for label not found in targets_transformed: {l} (label)!",
                                     "Update file containing all moves!")

            # transforming formatted data
            if self.transform:
                targets = self.transform(targets, grad=True)
                byteboards = self.transform(byteboards, grad=True)

            # detaching byteboards and labels
            byteboards.clone().detach()
            targets.clone().detach()
            # adding the targets and the labels to the transformed data
            self.transformed_labels.append(targets)
            self.transformed_games.append(byteboards)

            if idx % 1000 == 0:
                print(f'Successfully formatted {idx} out of {len(self.data)} training cases!')


def get_chunked_dataset(path: str, color: c.COLORS, rnn: bool):
    chunk_size = 10_000_000
    for chunk_data, chunk_labels in zip(pd.read_csv(path, usecols=range(12), chunksize=chunk_size), pd.read_csv(path, usecols=[12], chunksize=chunk_size)):
        yield ChessDataset(None, color, rnn, transform=dt.to_tensor, data=chunk_data, labels=chunk_labels)


def prepare_chess_data(path: str, rnn: bool):
    """
    This function will convert a csv file into a np.array()
    :param path: path to the CSV file
    :param rnn: determines what dataset to use for cnn / rnn training
    """
    if rnn is True:
        # change the base path based on where the CSV folder is located
        bitboards = np.array(pd.read_csv(path, usecols=[0]).map(ast.literal_eval))

        labels = np.array(pd.read_csv(path, usecols=[1]).map(ast.literal_eval))  # new method turns label into tensor
    else:
        # change the base path based on where the CSV folder is located
        bitboards = np.array(pd.read_csv(path, usecols=range(12)))

        labels = np.array(pd.read_csv(path, usecols=[12]))  # new method turns label into tensor

    return bitboards, labels


def init_chess_dataset(color: c.COLORS, rnn: bool) -> ChessDataset:
    if color is not True and color is not False:
        raise ValueError(f"Variable color must be of type {c.COLORS} but is of type {type(color)}!")

    if rnn is True:
        paths = None
    else:
        paths = [WHITE_GAMES_CSV, BLACK_GAMES_CSV]

    if color is True:
        dataset = ChessDataset(paths[0],
                               color,
                               rnn,
                               transform=dt.to_tensor)  # dt.RandomCrop(4) additionally
        return dataset

    elif color is False:
        dataset = ChessDataset(paths[1],
                               color,
                               rnn,
                               transform=dt.to_tensor)
        return dataset


def get_output_length(color: c.COLORS) -> int:
    """Getting the total number of learnable moves"""
    df = None

    if type(color) is not bool:
        raise ValueError(f"Expected type {c.COLORS} but received type {type(color)}")

    if color is True:
        df = pd.read_csv(WHITE_MOVES_CSV)
    if color is False:
        df = pd.read_csv(BLACK_MOVES_CSV)

    return df.__len__()


def file_exceeds_size(path: str, size: int) -> bool:
    """
    Check whether a given file exceeds a certain size
    Args:
        path: path to the file
        size: max size of the file in bytes

    Returns: True if file greater than size, else False

    """
    return os.path.getsize(path) > size
