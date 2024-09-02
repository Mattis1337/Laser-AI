import chess
import chess.pgn


# CONVERTING ANNOTATIONS
def fen_to_bitboards(fen: str) -> list[int]:
    """
    Converts a FEN to an array of integers. Converted to binary, they return bitboards.
    The order in the array is the same as python chess':
        0. White pawns
        1. White knights
        [...] sorted by value
        6. black pawns
        7. black knights
        [...] sorted by value
        11. black king

    Args:
        fen (str): board state in FEN to convert

    Returns:
        list[int]: list of 12 bitboards representing the board state
    """
    bitboards: list[int] = []

    try:
        board = chess.Board(fen)
    except ValueError:
        print("Invalid Forsyth Edwards Notation!")
        return bitboards

    for color in chess.COLORS:
        for piece_type in chess.PIECE_TYPES:
            # gets bitboard of piece type and color
            bitboard: int = board.pieces_mask(
                piece_type=piece_type,
                color=color
            )
            bitboards.append(bitboard)

    return bitboards


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


def pgn_to_bitboards_snapshots(pgn):
    """
    Converts a PGN file to two arrays. The first one contains bitboards grouped by piece type in a subarray.
    The next move is stored in second one as well. This is done by snapshotting the board after each move.
    Therefore, accessing the corresponding move to the bitboard can be done by using the same index
    :param pgn: The PGN file to read
    :return: Two arrays containing bitboards and the next move in SAN. NULLABLE!
    """
    try:
        game = chess.pgn.read_game(pgn)  # loads game in
        board = game.board()
    except ValueError:
        return None, None

    bitboards = []
    moves = []

    # iterates through every move
    for move in game.mainline_moves():
        bitboard = fen_to_bitboards(board.fen())    # gets current bitboard
        san = board.san(move).replace('+', '')  # converts move to san and removes data causing over-fitting
        # saves the data
        bitboards.append(bitboard)
        moves.append(san)
        # plays next move
        board.push(move)

    return bitboards, moves


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
