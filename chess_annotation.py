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

    for color in chess.COLORS:
        for piece_type in chess.PIECE_TYPES:
            # gets bitboard of piece type and color
            bitboard: int = board.pieces_mask(
                piece_type=piece_type,
                color=color
            )
            bitboards.append(bitboard)

    return bitboards


def fen_to_bitboards(fen: str) -> list[int]:
    try:
        board = chess.Board(fen)
    except ValueError:
        print("Invalid Forsyth Edwards Notation!")
        return []

    return get_bitboards(board)


# boards = fen_to_bitboards(chess.STARTING_FEN)
# for bb in boards:
#     print(format(bb, '064b'))


def pgn_to_bitboards_final(pgn):
    """
    Plays every move of a PGN file and saves the last board state as a bitboard.
    :param pgn: The PGN file to read
    :return: The final bitboard. NULLABLE!
    """
    try:
        game = chess.pgn.read_game(pgn)  # loads game in
        board = game.board()
    except ValueError:
        return None

    # iterates every move
    for move in game.mainline_moves():
        board.push(move)  # plays current move

    return fen_to_bitboards(board.fen())


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

    try:
        game = chess.pgn.read_game(pgn)  # loads game in
        board = game.board()
    except (IOError, OSError, ValueError):
        print("Invalid Portable Game Format!")
        return states, moves

    # iterates through every move
    for move in game.mainline_moves():
        # gets current bitboard
        bitboards: list[int] = get_bitboards(board)
        states.append(bitboards)
        # converts move to Universal Chess Interface
        uci: str = board.uci(move)
        moves.append(uci)
        # plays next move
        board.push(move)

    return states, moves


def print_bitboard(bitboard):
    """
    Prints a bitboard in human-readable format to standard output.
    :param bitboard: The bitboard to print
    """
    binary = format(bitboard, '064b')    # converts int to binary
    formatted = '\n'.join([binary[i:i + 8] for i in range(0, len(binary), 8)])  # inserts linebreaks representing rows
    print(formatted)


def print_type_bitboards(bitboards):
    """
    Prints every piece bitboard and the piece type it is representing to standard output in a human-readable format.
    :param bitboards: An array of 12 piece bitboards created by fen_to_bitboards
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
