# import libraries
import os

# calculations
import numpy as np

# pgn converter
import chess
import chess.pgn

#####PATHS TO DATA ETC######
path_all_moves = '/home/mattis/Documents/Jugend_Forscht_2023.24/all_moves.txt' # File containing all moves
path_data_folder = '/home/mattis/Documents/Jugend_Forscht_2023.24/chess_data/' # Folder containing all training data

# Dirs for saving training inputs -> TODO: THESE DIRS DONT EXIST YET AND USING THEM WILL CAUSE AN ERROR
path_white_moves = '/home/mattis/Documents/Jugend_Forscht_2023.24/finished_data/white_img'
path_white_labels = '/home/mattis/Documents/Jugend_Forscht_2023.24/finished_data/white_labels' 
path_black_moves = '/home/mattis/Documents/Jugend_Forscht_2023.24/finished_data/black_img'
path_black_labels = '/home/mattis/Documents/Jugend_Forscht_2023.24/finished_data/black_labels'


#####CONVERTING ANNOTATIONS#####

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

    #TODO: Maybe adjust the dimensions to fit the pytorch requirements eg. instead of a 12x64 dimensions take 8x(8x12)

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
                raise ValueError('Invalid Fencode!')

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
    all_labels = []

    for move in game.mainline_moves():
        
        all_labels.append(board.san(move)) # San has to be done before pushing the move 
        
        board.push(move) # Then push a move 
        
        all_moves.append(fen_to_bitboard(board.fen())) # Then append the newly created bitboard


    return all_moves, all_labels


def print_bitboard_fen(bitboard):
    """prints a bitboard deriving from a fen"""
    for i in range(12):
        for j in range(64):
            if (j % 8 == 0):
                print('')

            print(bitboard[i][j], end='')

        print('')


#####CREATING ALL POSSIBLE OUTPUTS#####

def move_filter(move):
    """filters a move for '\n"""
    return str.strip(move)


def create_outputs():
    """creates a file containing all possible output cases"""
    path_p = path_all_moves  # Path to personal file containing all moves
    path_f = path_data_folder # Path to the folder containing all the games
    directories = os.listdir(path_f)

    found = 0

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
                found = 0
                file = open(path_p, 'r+')
                lines = file.readlines()
                for line in lines:
                    if p_move == move_filter(line):
                        found = 1
                        break
                if found != 1:
                    file.write(p_move + '\n')


#####INITIALISING FILES FOR INPUTS#####

def create_input_datasets():
    """
    creates 4 different files containing
    - all black moves as bitboards 
    - all black labels for each move as a notation (aka wanted output for the nn)
    - all white moves as bitboards 
    - all white labels for each move (aka wanted output for the nn)

    NOTE: It is to the utmost importance that these files are synchronized meaning 
          the according labels to a move have to be at the exact same index in their respective file
          as the fitting bitboard 
    """

    path_f = path_data_folder
    directories = os.listdir(path_f)

    for file in directories:
        # Main loop iterating through all the files
        file = open(path_f + file, 'r+')
        

        p_bitboard, p_labels = pgn_to_bitboard(file)

        #white_img = []
        #white_labels = []
        #black_img = []
        #black_labels = []

        for i in range(len(p_labels)):
            # len(p_bitboard)-1 = len(p_labels) therefore last position is ignored ()

            if i % 2 == 0:
                with open(path_white_moves , 'r+') as file_w:
                    for j in p_bitboard[i]:
                        file_w.write(str(j))
                    file.write('\n')

                with open(path_white_labels , 'r+') as file_w:
                    print(p_labels[i])
                    string = str(p_labels[i])+'\n'
                    file_w.write(string)
            
            else:
                with open(path_black_moves , 'r+') as file_w:
                    for j in p_bitboard[i]:
                        file_w.write(str(j))
                    file.write('\n')

                with open(path_black_labels , 'r+') as file_w:
                    print(p_labels[i])
                    string = str(p_labels[i])+'\n'                    
                    file_w.write(string)


#TODO: Create input and create output move to csv branch and data_preparer file and will create csv files
