import chess
import chess.pgn

# PATHS TO DATA ETC
path_all_moves = '/home/mattis/Documents/Jugend_Forscht_2023.24/all_moves.txt'  # File containing all moves
path_data_folder = '/home/mattis/Documents/Jugend_Forscht_2023.24/chess_data/'  # Folder containing all training data

# Dirs for saving training inputs -> TODO: THESE DIRS DONT EXIST YET AND USING THEM WILL CAUSE AN ERROR
path_white_moves = '/home/mattis/Documents/Jugend_Forscht_2023.24/finished_data/white_img'
path_white_labels = '/home/mattis/Documents/Jugend_Forscht_2023.24/finished_data/white_labels' 
path_black_moves = '/home/mattis/Documents/Jugend_Forscht_2023.24/finished_data/black_img'
path_black_labels = '/home/mattis/Documents/Jugend_Forscht_2023.24/finished_data/black_labels'


# CONVERTING ANNOTATIONS
def fen_to_bitboards(fen):
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
    :param fen: The notation to convert
    :return: An array of twelve bitboards. NULLABLE!
    """
    try:
        board = chess.Board(fen)
    except ValueError:
        return None

    bitboards = []

    for color in chess.COLORS:
        for piece_type in chess.PIECE_TYPES:
            # creates an empty bitboard for the specific type of piece and exact color
            bitboard = chess.BB_EMPTY
            piece = chess.Piece(piece_type, color)

            # iterates over all squares
            for square in chess.SQUARES:
                if board.piece_at(square) == piece:
                    bitboard |= 1 << square  # sets the square-th bit to 1

            # adds the bitboard to the list
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
        # saves the data
        bitboards.append(bitboard)
        moves.append(move)
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


# TODO: Create input and create output move to csv branch and data_preparer file and will create csv files
