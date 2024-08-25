import requests
import time
import chess


def request_ai_move(url: str, fen: str):
    payload = {
        "fen": fen,
    }
    response = requests.get(url, json=payload)
    if response.status_code != 200:
        print(f"Request failed with code {response.status_code}! Retrying...")
        time.sleep(1)
        return request_ai_move(url, fen)
    
    return response.json()["move"]


def request_user_move(board: chess.Board):
    move = input("Insert your next move: ")
    if not move in board.legal_moves:
        print("Invalid move")
        return request_user_move(board)
    return move
