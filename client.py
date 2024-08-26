import requests
import time
import chess


def request_ai_move(url: str, fen: str, retries: int):
    payload = {
        "fen": fen,
    }

    while retries > 0:
        retries -= 1

        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(f"Request failed with code {response.status_code}! Retrying...")
            time.sleep(1)
            continue

        try:
            move = response.json()["move"]
        except (requests.exceptions.JSONDecodeError, KeyError):
            print(f"Couldn't decode JSON received by server! Retrying...")
            continue

        return move

    return None


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
