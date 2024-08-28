import client
import chess


def play_against_ai(fen: str, unicode: bool, ai_host: str, ai_color: chess.Color):

    try:
        board = chess.Board(fen)
    except ValueError:
        print("Invalid FEN code provided in command arguments!")
        exit(1)

    print (
        "\n" +
        "Please express the moves you want to play " +
        "by writing down the square of the piece you want to move " +
        "then followed by the square it should be moved to (e.g. 'c2c4')!",
        "\nHint: That notation is called Universal Chess Interface (UCI)"
    )

    while not board.is_game_over():
        # print beautiful CLI chess board
        print_board(board, unicode, side=not ai_color)
        # generate move
        if board.turn is ai_color:
            move = client.request_ai_move(board, url=ai_host, retries=5)
            if move is None:
                print("AI couldn't generate move! Is the server set up properly?")
                break
        else:
            move = client.request_user_move(board)
        # play move
        board.push(move)
    print(f"Final board state is {board.fen()}")


def print_board(board: chess.Board, unicode: bool, side: chess.Color):
    # print board with ASCII characters
    if not unicode:
        print('-' * 8 * 2)
        print(board)
        return
    # print board with unicode characters
    print(u'\u2500' * 8 * 2)
    print(board.unicode(
        invert_color=True,
        #borders=True,
        empty_square=".",
        orientation=side,
    ))
