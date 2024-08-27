import requests
import time
import chess


def request_ai_move(board: chess.Board, url: str, retries: int = 5, delay: int = 1):
    payload = {
        "fen": board.fen(),
    }

    while retries > 0:
        retries -= 1

        try:
            response = requests.post(url, json=payload)
        except requests.ConnectionError as error:
            print(f"{error}! Retrying...")
            time.sleep(delay)
            continue

        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            print(f"{error}! Retrying...")
            time.sleep(delay)
            continue

        try:
            move_san: str = response.json()["move"]
        except (requests.JSONDecodeError, KeyError):
            print("Couldn't decode JSON sent by server! Retrying...")
            time.sleep(delay)
            continue

        try:
            move = board.parse_san(move_san)
        except ValueError:
            print("AI used invalid SAN notation for move!")
            time.sleep(delay)
            continue

        if not board.is_legal(move):
            print("Move generated by AI is illegal!")
            time.sleep(delay)
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

        if not board.is_legal(move):
            print("Illegal move!")
            continue

        return move
