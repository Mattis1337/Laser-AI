import client
import chess


def play_against_ai(ai_color: chess.Color, url: str):
    print (
        "Please express the moves you want to play \
        by stating the square of the piece you want to move and the square \
        you wan to move to e.g. 'c2c4'!"
    )
    board = chess.Board()
    while not board.is_game_over:
        print(board)
        if board.turn is ai_color:
            move = request_ai_move(url, board.fen)
        else:
            request_user_move(board)
        board.push(move)
