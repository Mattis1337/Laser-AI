# import libraries
import io
# TODO: temp
import os

# calculations
import numpy as np

# pgn converter
import chess
import chess.pgn


def fen_to_bitboard(fencode):
    """
    Function to transform FEN notation into Bitboard notation
    :param fencode: a fencode as a  string
    :return: a bitboard[12][64]

    WHITE PIECES

    bitboard[0] : white king (K)
    bitboard[1] : white queen (Q)
    bitboard[2] : white rook (R)
    bitboard[3] : white knight (N)
    bitboard[4] : white bishop (B)
    bitboard[5] : white pawn (P)

    BLACK PIECES

    bitboard[6] : black king (k)
    bitboard[7] : black queen (q)
    bitboard[8] : black rook (r)
    bitboard[9] : black knight (n)
    bitboard[10] : black bishop (b)
    bitboard[11] : black pawn (p)
    """

    # Initialising multidimensional array as bitboard
    bitboard = np.full(shape=(12, 64), fill_value=0)
    # field serves as a counter for the chess field one is operating
    field = 0

    for i in range(len(fencode)):

        n = fencode[i]

        if n.isdigit():
            field += int(n)
            continue

        if n == '/':
            continue

        if n == ' ':
            # checking whether this is the end of the notation or not
            return bitboard

        match n:
            case 'K':
                bitboard[0][field] = 1
                field += 1

            case 'Q':
                bitboard[1][field] = 1
                field += 1

            case 'R':
                bitboard[2][field] = 1
                field += 1

            case 'N':
                bitboard[3][field] = 1
                field += 1

            case 'B':
                bitboard[4][field] = 1
                field += 1

            case 'P':
                bitboard[5][field] = 1
                field += 1

            case 'k':
                bitboard[6][field] = 1
                field += 1

            case 'q':
                bitboard[7][field] = 1
                field += 1

            case 'r':
                bitboard[8][field] = 1
                field += 1

            case 'n':
                bitboard[9][field] = 1
                field += 1

            case 'b':
                bitboard[10][field] = 1
                field += 1

            case 'p':
                bitboard[11][field] = 1
                field += 1

            case _:
                return "INVALID FENCODE"

    return bitboard


def pgn_to_bitboard(file):
    """
    Function to transform the algebraic chess notation into a bitboard.
    :param file: the file containing the notation
    :return: a bitboard displaying the chess field according to the notation
    """

    game = chess.pgn.read_game(file)

    # function will have to receive:
    # game; must be a variable containing the whole game like presented above (from chess library)
    board = game.board()
    all_moves = [fen_to_bitboard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR')]

    for move in game.mainline_moves():
        board.push(move)
        all_moves.append(fen_to_bitboard(board.fen()))

    return all_moves


def print_bitboard_fen(bitboard):
    for i in range(12):
        for j in range(64):
            if (j % 8 == 0):
                print('')

            print(bitboard[i][j], end='')

        print('')


# Filters move notation for '\n'
def move_filter(move):
    return str.strip(move)


# Scans the file while dealing with the increasing amount of lines there are
def scan_file(path, p_move):
    with open(path, 'r+') as file:
        lines = file.readlines()
        for line in lines:
            if p_move == move_filter(line):
                return

        print(p_move)
        print(move_filter(line))
        file.write(p_move + '\n')


def create_outputs():
    path_p = 'C:/Users/frank\OneDrive\Desktop/allMoves.txt'  # Path to personal file containing all moves
    path_f = 'C:/Users/frank\OneDrive\Desktop\chess_game'  # Path to the folder containing all the games
    directories = os.listdir(path_f)

    for file in directories:
        f_current = open(path_f + '/' + file, 'r')
        game = chess.pgn.read_game(f_current)  # Opens current game from databank
        board = chess.Board(chess.STARTING_BOARD_FEN)  # Creates board all moves will be pushed from

        for move in game.mainline_moves():  # Add move to board to get diff situation
            board.push(move)

            legal_moves_lst = [
                board.san(move)
                for move in board.legal_moves
            ]

            for p_move in legal_moves_lst:
                scan_file(path_p, p_move)
