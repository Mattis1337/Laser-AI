import numpy as np

print("Laser-AI")

def closest(x, values):
    """
    This function will return the closest value/ number to a given value
    :param x: (random) value that is to be inspected
    :param values: list of values x is to be compared to
    :return: value closest to x
    """

    #TODO: This function need to be added to the actual network.

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



    bitboard = np.full(shape=(12, 64), fill_value=0)
    print(bitboard[1][1])

    for i in range(fencode.length):


        #TODO: Actually test this there might be some bugs or errors but its getting late...

        field = 0

        if fencode[i].type == int:
            # add the numbers of free fields to var field
            field += fencode[i]
            continue

        if fencode[i] == '/':
            continue

        if fencode[i] == ' ':
            # checking whether this is the end of the notation or not
            return bitboard

        match fencode[i]:
            case 'K':
                bitboard[0][field] = 1
                field+=1
                pass
            case 'Q':
                bitboard[1][field] = 1
                field += 1
                pass
            case 'R':
                bitboard[2][field] = 1
                field += 1
                pass
            case 'N':
                bitboard[3][field] = 1
                field += 1
                pass
            case 'B':
                bitboard[4][field] = 1
                field += 1
                pass
            case 'P':
                bitboard[5][field] = 1
                field += 1
                pass
            case 'k':
                bitboard[6][field] = 1
                field += 1
                pass
            case 'q':
                bitboard[7][field] = 1
                field += 1
                pass
            case 'r':
                bitboard[8][field] = 1
                field += 1
                pass
            case 'n':
                bitboard[9][field] = 1
                field += 1
                pass
            case 'b':
                bitboard[10][field] = 1
                field += 1
                pass
            case 'p':
                bitboard[11][field] = 1
                field += 1
                pass
            case _:
                return "INVALID FENCODE"

