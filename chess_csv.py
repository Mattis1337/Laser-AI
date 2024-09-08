import os
import glob
from typing import Iterator
import itertools
import pandas as pd

import chess_annotation as annotation

# the directory containing chess game representations in PGN format
pgn_dir = r"Games"
os.makedirs(pgn_dir, exist_ok=True)

# your local directory containing the CSV/ folder
csv_dir = r"CSV"
os.makedirs(csv_dir, exist_ok=True)
# the paths to save the training data to
white_games_csv: str = os.path.join(csv_dir, r"white_games.csv")
black_games_csv: str = os.path.join(csv_dir, r"black_games.csv")
# the paths to save the outputs to
white_moves_csv: str = os.path.join(csv_dir, r"white_moves.csv")
black_moves_csv: str = os.path.join(csv_dir, r"black_moves.csv")


def get_pgn_paths(directory: str, chunks: int = 1) -> Iterator[tuple[str]]:
    """
    Gets all the fs paths to PGN files that should be read and converted to bitboards.
    Then they are partitioned into multiple small arrays. 

    Args:
        directory (str): The folder containing the PGNs to scan
        chunks (int): Natural number above 0 and less than the amount of files in the directory
            that represents the amount of subarrays to create

    Raises:
        FileNotFoundError: glob doesn't raise an error if the directory is empty,
            but there is no need to execute the script any further without PGN files

    Returns:
        Iterator[tuple[str]]: A list that contains tuples of evenly distributed PGN files in the target directory
    """
    # gets all the paths to files that end with .pgn
    pgn_file_paths: list[str] = glob.glob(os.path.join(directory, "*.pgn"))

    if not pgn_file_paths:
        raise FileNotFoundError(f"No PGN files found in {directory}.")

    if chunks < 1:
        print("'chunks' must be a natural number and not 0 because the resulting array can't be divided by 0!")
        chunks = 1

    if chunks > len(pgn_file_paths):
        print(f"'chunks' must be smaller than the amount of files in {directory}.")
        chunks = len(pgn_file_paths)

    # slices list into smaller chuncks
    return itertools.batched(pgn_file_paths, chunks)


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

            # every bitboard has its own column
            data = {
                'bitboards_wP': state[0],
                'bitboards_wN': state[1],
                'bitboards_wB': state[2],
                'bitboards_wR': state[3],
                'bitboards_wQ': state[4],
                'bitboards_wK': state[5],
                'bitboards_bP': state[6],
                'bitboards_bN': state[7],
                'bitboards_bB': state[8],
                'bitboards_bR': state[9],
                'bitboards_bQ': state[10],
                'bitboards_bK': state[11],
                'move': moves[i],
            }

            # white always plays even move number: 0 (first play), 2, 4, [...]
            if i % 2 == 0:
                white_data.append(data)
            else:
                black_data.append(data)

    return white_data, black_data


def write_csv(data, path: str):
    """
    Writes data to a CSV file using this project's default format.
    :param data: The data to write
    :param path: The path of the CSV file to write to
    """
    df = pd.DataFrame(data)
    df.to_csv(path, mode="a", header=False, index=False)


def convert_multiple_pgns_to_csv(pgn_file_paths: list[str], white_games_path: str, black_games_path: str):
    """
    Converts a list of PGN files to bitboards and saves them in two separate CSVs;
    One for black's moves and one for white's.
    :param pgn_file_paths: The PGNs to parse
    :param white_games_path: The location of the white side's CSV
    :param black_games_path: The location of the black side's CSV
    """
    for path in pgn_file_paths:  # iterates through every file
        # converts PGN to CSV
        white_data, black_data = convert_single_pgn_to_csv(pgn_path=path)
        # writes the data
        write_csv(data=white_data, path=white_games_path)
        write_csv(data=black_data, path=black_games_path)
        # debug information
        print(f"[CSV] Wrote data from {path}!")


def create_one_output(game_csv: str, save_path: str):
    """
    Gets all the moves written to the second column of one color's dataset and removes all duplicate moves.
    This will ensure that no under-fitting will occur as a result of the AI's outputs,
    as each one is only present once.
    :param game_csv: A CSV file containing mappings of board states and moves, created by convert_png_to_csv()
    :param save_path: The path where the new CSV file should be saved at
    """
    games = pd.read_csv(game_csv)
    moves = games.iloc[:, -1].drop_duplicates()  # gets the last column and removes duplicate moves
    moves.to_csv(save_path, header=False, index=False)


def create_csvs():
    # annotation.pgn_to_bitboards_snapshots()
    print(get_pgn_paths(pgn_dir))
    convert_multiple_pgns_to_csv(
       pgn_file_paths=get_pgn_paths(directory=pgn_dir),
       white_games_path=white_games_csv,
       black_games_path=black_games_csv
    )
    create_one_output(game_csv=white_games_csv, save_path=white_moves_csv)
    print(f"[CSV] Created white outputs successfully in {white_moves_csv}")
    create_one_output(game_csv=black_games_csv, save_path=black_moves_csv)
    print(f"[CSV] Created black outputs successfully in {black_moves_csv}")


if __name__ == "__main__":
    create_csvs()
