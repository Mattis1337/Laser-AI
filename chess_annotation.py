import chess
import chess.pgn


# CONVERTING ANNOTATIONS
def get_bitboards(board: chess.Board) -> list[int]:
    """
    Converts board instance to an array of integers. Converted to binary, they return bitboards.
    The order in the array is the same as python chess':
        0. White pawns
        1. White knights
        [...] sorted by value
        6. black pawns
        7. black knights
        [...] sorted by value
        11. black king

    Args:
        board (chess.Board): the board to represent as a bitboard

    Returns:
        list[int]: list of 12 bitboards representing the board state
    """
    bitboards: list[int] = []

    if board is None or type(board) is not chess.Board:
        print("chess.Board instance must be provided!")
        return bitboards

    for color in chess.COLORS:
        for piece_type in chess.PIECE_TYPES:
            # gets bitboard of piece type and color
            bitboard: int = board.pieces_mask(
                piece_type,
                color,
            )
            bitboards.append(bitboard)

    return bitboards


def fen_to_bitboards(fen: str) -> list[int]:
    """
    Parses a FEN code to a chess.Board instance and then generates the bitboards.
    See also: get_bitboards(chess.Board)

    Args:
        fen (str): The FEN code to convert

    Returns:
        list[int]: A list containing the 12 bitboards
    """
    try:
        board = chess.Board(fen)
    except ValueError:
        print("Invalid Forsyth Edwards Notation!")
        return []

    return get_bitboards(board)


# boards = fen_to_bitboards(chess.STARTING_FEN)
# for bb in boards:
#     print(format(bb, '064b'))


def pgn_to_bitboards_final(pgn: TextIO) -> list[int]:
    """
    Plays every move of a PGN file and saves the last board state as a bitboard.
    See also: get_bitboards(chess.Board)

    Args:
        pgn (TextIO): The handle of the PGN file to read

    Returns:
        list[int]: The final board state represented by 12 bitboards.
    """
    # loads game in
    try:
        game = chess.pgn.read_game(pgn)
    except (OSError, IOError, FileNotFoundError) as error:
        print("PGN file couldn't be read! " + error)
        return []

    if game is None:
        print("Invalid Portable Game Format!")
        return []

    board = game.board()

    # iterates every move
    for move in game.mainline_moves():
        board.push(move)  # plays current move

    return get_bitboards(board)


def pgn_to_bitboards_snapshots(pgn: TextIO) -> (list[list[int]], list[str]):
    """
    Converts a PGN file to two arrays. The first one contains bitboards grouped by piece type in a subarray.
    The next move is stored in second one as well. This is done by snapshotting the board after each move.
    Therefore, accessing the corresponding move to the bitboard can be done by using the same index

    Args:
        pgn (TextIO): handle to the PGN file to read

    Returns:
        list[list[int]]: each element of the first dimension is a board state. 
            The nested lists represent the bitboards of each state.
        list[str]: the move played for the board state in a 1:1 mapping
    """
    states: list[list[int]] = []
    moves: list[str] = []

    # loads game in
    try:
        game = chess.pgn.read_game(pgn)
    except (OSError, IOError, FileNotFoundError) as error:
        print("PGN file couldn't be read! " + error)
        return states, moves

    if game is None:
        print("Invalid Portable Game Format!")
        return states, moves

    board = game.board()    

    # iterates through every move
    for move in game.mainline_moves():
        bitboards: list[int] = get_bitboards(board)
        # converts move to Universal Chess Interface
        uci: str = board.uci(move)

        # plays next move if legal
        if not board.is_legal(move):
            print("Illegal move in PGN file!")
            break
        board.push(move)

        states.append(bitboards)
        moves.append(uci)

    return states, moves


def print_bitboard(bitboard: int):
    """
    Prints a bitboard in human-readable format to standard output.

    Args:
        bitboard (int): The bitboard to print
    """
    binary = format(bitboard, '064b')    # converts int to binary
    formatted = '\n'.join([binary[i:i + 8] for i in range(0, len(binary), 8)])  # inserts linebreaks representing rows
    print(formatted)


def print_bitboards(bitboards: list[int]):
    """
    Prints the bitboards returned by the get_bitboards() function and the piece type it is representing
    to standard output in a human-readable format.

    Args:
        bitboards (list[int]): An array of 12 piece bitboards created by fen_to_bitboards
    """
    for i, color in enumerate(chess.COLOR_NAMES[::-1]):  # slicing reverses list because COLOR_NAMES is mirrored
        for j, piece in enumerate(chess.PIECE_NAMES[1:]):   # slicing removes first element which would be None
            print(f"\n {color} {piece}s:")
            # calculates the index in the bitboards array corresponding to the current color and piece type
            index = i * len(chess.PIECE_TYPES) + j
            print_bitboard(bitboards[index])


# CREATING ALL POSSIBLE OUTPUTS

def move_filter(move):
    """filters a move for '\n"""
    return str.strip(move)


def bitboard_to_byteboard(bitboard: int) -> list[int]:
    return [int(bit) for bit in format(int(bitboard), '064b')]  # formats int to str(bits) to list[int]
