import os
from glob import glob
import pandas as pd

import chess_annotation as annotation


# a directory containing PGN files
games_dir = r"Games"
# the files to save the moves to
white_moves_csv = r"CSV/white_moves.csv"
black_moves_csv = r"CSV/black_moves.csv"


def get_pgn_paths(directory: str) -> list[str]:
    """
    Gets all the fs paths to PGN files that should be read and converted to bitboards.
    :param directory: The folder containing the PGNs to scan
    :return: A list of all PGN files in the game_dir
    """
    return glob(os.path.join(directory, "*.pgn"))


def convert_single_pgn_to_csv(pgn_path: str) -> tuple[list, list]:
    """
    Converts a PGN file's moves into board states mapped to the next move.
    Then it converts them into memory-efficient bitboards.
    Lastly, the data is saved in two csv data sets (IN RAM) representing white and black moves respectively.
    :param pgn_path: The PGN file to load
    :return: Two lists of CSV data representing white and black board states/moves
    """
    white_data = []
    black_data = []

    # loads PGN file
    with open(pgn_path, "r") as pgn:
        board_states, moves = annotation.pgn_to_bitboards_snapshots(pgn)

        # iterates over board_states which are represented by 12 bitboards in a list
        for i, state in enumerate(board_states):
            # prevents crash, if bitboards couldn't be loaded
            if state is None or moves[i] is None:
                continue
            if any(bitboard is None for bitboard in state):
                continue

            data = {
                'bitboards': state,
                'move': moves[i],
            }

            # white always plays even move number: 0 (first play), 2, 4, [...]
            if i % 2 == 0:
                white_data.append(data)
            else:
                black_data.append(data)

    return white_data, black_data


def write_csv(data: list, path: str):
    """
    Writes data to a CSV file using this project's default format.
    :param data: The data to write
    :param path: The path of the CSV file to write to
    """
    df = pd.DataFrame(data)
    df.to_csv(path, mode="a", header=False, index=False)


def convert_multiple_pgns_to_csv(pgn_file_paths: list[str], white_csv: str, black_csv: str):
    """
    Converts a list of PGN files to bitboards and saves them in two separate CSVs;
    One for black's moves and one for white's.
    :param pgn_file_paths: The PGNs to parse
    :param white_csv: The location of the white side's CSV
    :param black_csv: The location of the black side's CSV
    """
    for path in pgn_file_paths:  # iterates through every file
        # converts PGN to CSV
        white_data, black_data = convert_single_pgn_to_csv(path)
        # writes the data
        write_csv(white_data, white_csv)
        write_csv(black_data, black_csv)
        # debug information
        print(f"[CSV] Wrote data from {path}!")


# annotation.pgn_to_bitboards_snapshots()
# print(get_pgn_paths(games_dir))
convert_multiple_pgns_to_csv(get_pgn_paths(games_dir), white_moves_csv, black_moves_csv)
