import requests
import time
import chess


def request_ai_move(url: str, fen: str):
    payload = {
        "fen": fen,
    }
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        print(f"Request failed with code {response.status_code}! Retrying...")
        time.sleep(1)
        return request_ai_move(url, fen)
    
    return response.json()["move"]


def request_user_move(board: chess.Board):
    while True:
        move = input("Insert your next move: ")
        try:
            move = board.parse_uci(move)
        except ValueError:
            print("Invalid UCI notation of move!")
            continue

        if move not in board.legal_moves:
            print("Illegal move")
            continue

        return move
