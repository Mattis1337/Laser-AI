# import libraries
import numpy as np
from enum import Enum
import os

def closest(x, values):
    """
    This function will return the closest value/ number to a given value
    :param x: (random) value that is to be inspected
    :param values: list of values x is to be compared to
    :return: value closest to x
    """

    for i, value in enumerate(values):

        if i == 0:
            closest = value
            p_diff = abs(value-x)   # p_diff is the closest difference measured yet
            continue

        diff = abs(value - x)   # diff is the difference of the current value to x

        if diff < p_diff:
            closest = value
            p_diff = diff

    return closest


# Another function determining which number is closer to a wanted value
# in case there is for instance

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

        if fencode[i].isdigit():
            # add the numbers of free fields to var field
            field += int(fencode[i])
            continue

        if fencode[i] == '/':
            continue

        if fencode[i] == ' ':
            # checking whether this is the end of the notation or not
            return bitboard

        rand = fencode[i]

        match rand:
            case "K":
                bitboard[0][field] = 1
                field = field + 1
                print("King at {}".format(field))
            case "Q":
                bitboard[1][field] = 1
                field += 1

            case "R":
                bitboard[2][field] = 1
                field += 1

            case "N":
                bitboard[3][field] = 1
                field += 1

            case "B":
                bitboard[4][field] = 1
                field += 1

            case "P":
                bitboard[5][field] = 1
                field += 1

            case "k":
                bitboard[6][field] = 1
                field += 1

            case "q":
                bitboard[7][field] = 1
                field += 1

            case "r":
                bitboard[8][field] = 1
                field += 1

            case "n":
                bitboard[9][field] = 1
                field += 1

            case "b":
                bitboard[10][field] = 1
                field += 1

            case "p":
                bitboard[11][field] = 1
                field += 1

            case _:
                return "INVALID FENCODE"


class field_note(Enum):
    """
    Enums regarding the field notations
    """

    # The notation is set from the white pov
    # in the bitboard the white side will be the downward facing one
    a = 0
    b = 1
    c = 2
    d = 3
    e = 4
    f = 5
    g = 6
    h = 7




def algebraic_to_bitboard(notation):
    """
    Function to transform the algebraic chess notation into a bitboard.
    :param notation: string containing the notation
    :return: a bitboard displaying the chess field according to the notation
    """

    # Initialising the bitboard
    bitboard = np.full(shape=(12,8,8), fill_value=0)

    state = 0
    number = 0

    for n in notation:
        if state != 1:
            if n != '1':
                continue
            if notation[n+1] != '.':
                continue
            state = 1




def print_bitboard(bitboard):
    for i in range(12):
        for j in range(64):
            if (j % 8 == 0):
                print('')

            print(bitboard[i][j], end='')

        print("")



######################################TESTING SITE##################################################################

# TODO: This is how to check whether an argument given by the string is a field or not and if
#   so the value of field can be used for placing the object in the matrix

class field(Enum):
    t = 0
    b = 1

string = "b"


list = [2.5,5,8]
for fs in field:
    print(fs.name)
    if string == fs.name:
        print(list[fs.value])



# TODO: Need to implement this when the actual files start appearing X)

fi = open("app.py", "r")
print(fi.read())

notation = fi.read()

print(notation)
