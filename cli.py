import client
import chess


def play_against_ai(ai_color: chess.Color, url: str, fen: str = chess.STARTING_FEN):
    print (
        "Please express the moves you want to play " +
        "by stating the square of the piece you want to move and the square " +
        "you want to move to e.g. 'c2c4'!"
    )
    board = chess.Board(fen)
    while not board.is_game_over():
        print(board)
        if board.turn is ai_color:
            move = client.request_ai_move(url, board, retries=5)
            if move is None:
                print("AI couldn't generate move! Is the server set up properly?")
                break
        else:
            move = client.request_user_move(board)
        board.push(move)
