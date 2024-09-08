# managing files
import os
import glob
import shutil
from typing import Iterator
import itertools
# async
from multiprocessing import Process
import asyncio
from concurrent.futures import ThreadPoolExecutor
# parsing data
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


def get_pgn_paths(directory: str, chunk_amount: int = 1) -> Iterator[tuple[str]]:
    """
    Gets all the fs paths to PGN files that should be read and converted to bitboards.
    Then they are partitioned into multiple small arrays. 

    Args:
        directory (str): The folder containing the PGNs to scan
        chunk_amount (int): Natural number above 0 and less than the amount of files in the directory
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

    if chunk_amount < 1:
        print("'chunk_amount' must be a natural number and not 0 because the resulting array can't be divided by 0!")
        chunk_amount = 1

    if chunk_amount > len(pgn_file_paths):
        print(f"'chunk_amount' must be smaller than the amount of files in {directory}.")
        chunk_amount = len(pgn_file_paths)

    # slices list into chunk_amount sections
    chunk_size = len(pgn_file_paths) // chunk_amount
    return itertools.batched(pgn_file_paths, chunk_size)


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


def convert_multiple_pgns_to_csv(
    pgn_file_paths: tuple[list],
    white_games_path: str,
    black_games_path: str,
):
    for path in pgn_file_paths:
        white_data, black_data = convert_single_pgn_to_csv(pgn_path=path)
        # writes the data
        write_csv(data=white_data, path=white_games_path)
        write_csv(data=black_data, path=black_games_path)
        # debug information
        print(f"[CSV] Wrote data from {path}!")


def merge_multiple_files(file_paths: list[str], result_path: str, override_clone: bool = False, delete_orig: bool = False):
    """
    Merges multiple files into another one no matter their encoding.
    It can optionally delete the original files.

    Args:
        file_paths (list[str]): Paths to the files to merge the contents of
        result_path (str): The file to write the result to
        override_clone (bool): Whether to delete the contents of the target file if it wasn't empty. Defaults to False.
        delete_orig (bool, optional): Whether to delete the original files or not. Defaults to False.
    """
    write_mode = 'ab'
    if override_clone:
        write_mode = 'wb'

    with open(file=result_path, mode=write_mode) as result_file:
        for path in file_paths:
            # prevent errors
            if not os.path.exists(path):
                continue
            # concatenation
            try:
                with open(path,'rb') as input_file:
                    shutil.copyfileobj(input_file, result_file)
            except (IOError, OSError) as error:
                print(f"File at {path} coudln't be read or merged: {error}")
            # delete originals if toggled
            if delete_orig:
                os.remove(path)


async def all_pgns_to_csv(
    pgn_file_paths: Iterator[tuple[str]],
    white_games_path: str,
    black_games_path: str,
    merge_old_csv: bool = False,
    delete_csv_fragments: bool = True,
):
    """
    Converts a list of PGN files to bitboards and saves them in two separate CSVs;
    One for black's moves and one for white's.
    :param pgn_file_paths: The PGNs to parse
    :param white_games_path: The location of the white side's CSV
    :param black_games_path: The location of the black side's CSV
    """
    processes: list[Process] = []
    for i, chunk in enumerate(pgn_file_paths):
        # creates a process and runs it on a seperate physical thread
        process = Process(target=convert_multiple_pgns_to_csv, args=(
            chunk,
            # makes sure that no data is overriden because of race conditions
            f"{white_games_path}-part{i}",
            f"{black_games_path}-part{i}",
        ))
        process.start()
        processes.append(process)

    # syncs process to prepare for merging process
    for process in processes:
        process.join()

    # define preset for merging files on separate thread
    async def run_merge(games_path: str):
        await asyncio.to_thread(merge_multiple_files,
            glob.glob(f"{games_path}-part*"),
            games_path,
            not merge_old_csv,
            delete_csv_fragments,
        )

    # run threads
    threads = [
        run_merge(games_path=white_games_path),
        run_merge(games_path=black_games_path),
    ]

    # await threads to finish
    await asyncio.gather(*threads)


def create_output(game_csv: str, save_path: str):
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


def main():
    # annotation.pgn_to_bitboards_snapshots()
    pgn_files: Iterator[tuple[str]] = get_pgn_paths(
        pgn_dir,
        chunk_amount=os.cpu_count()
    )

    asyncio.run(all_pgns_to_csv(
       pgn_file_paths=pgn_files,
       white_games_path=white_games_csv,
       black_games_path=black_games_csv,
       merge_old_csv=False,
       delete_csv_fragments=True,
    ))

    create_output(game_csv=white_games_csv, save_path=white_moves_csv)
    print(f"[CSV] Created white outputs successfully in {white_moves_csv}")
    create_output(game_csv=black_games_csv, save_path=black_moves_csv)
    print(f"[CSV] Created black outputs successfully in {black_moves_csv}")


if __name__ == "__main__":
    main()
