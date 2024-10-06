import client
import chess
import atexit


BOARD: chess.Board


def play_against_ai(fen: str, unicode: bool, ai_host: str, ai_color: chess.Color):
    """
    Defines the gameplay loop. First loads the chess board using the provided FEN and verifies it's legal.
    Then a move is played by the AI or player in rotating turns. This continues until a win codition is met.

    Args:
        fen (str): Starting position of board
        unicode (bool): Whether to print Unicode chars
        ai_host (str): URL of AI server
        ai_color (chess.Color): Side of AI
    """
    # registering exit handler
    atexit.register(exit_handler)
    try:
        global BOARD
        BOARD = chess.Board(fen)
    except ValueError:
        print("Invalid FEN code provided in command arguments!")
        exit(1)

    print(
        "\n" +
        "Please express the moves you want to play " +
        "by writing down the square of the piece you want to move " +
        "then followed by the square it should be moved to (e.g. 'c2c4')!",
        "\nHint: That notation is called Universal Chess Interface (UCI)"
    )

    while not BOARD.is_game_over():
        # print beautiful CLI chess board
        print_board(BOARD, unicode, side=not ai_color)
        # generate move
        if BOARD.turn is ai_color:
            move = client.request_ai_move(BOARD, url=ai_host, retries=5)
            if move is None:
                print("AI couldn't generate move! Is the server set up properly?")
                break
        else:
            move = client.request_user_move(BOARD)
        # play move
        BOARD.push(move)


def print_board(board: chess.Board, unicode: bool, side: chess.Color):
    """
    Print an instance of a chess board.
    The difference to plain print(chess.Board) is that the function can print
    the chess board using Unicode characters if enabled by the user, but
    can also fallback to ASCII. There's also the ability to change the orientation,
    which is annoying to do manually on print(chess.Board).

    Args:
        board (chess.Board): Board to print
        unicode (bool): Whether to use Unicode or not
        side (chess.Color): The orientation the player is facing the board
    """
    # print board with ASCII characters
    if not unicode:
        print('-' * 8 * 2)
        # flip board if Black is the player
        print(
            board if side is chess.WHITE
            else board.transform(chess.flip_vertical).transform(chess.flip_horizontal)
        )
        return
    # print board with unicode characters
    print('\u2500' * 8 * 2)
    print(board.unicode(
        invert_color=True,
        #borders=True,
        empty_square=".",
        orientation=side,
    ))


def exit_handler():
    if BOARD is not None:
        print(f"Final board state is '{BOARD.fen()}'")
