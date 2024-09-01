import requests
import time
import chess


def request_ai_move(board: chess.Board, url: str, retries: int = 5, delay: int = 1):
    """
    Request a move from a remote host running the AI model as the REST API specipifications defined.
    A lot of error handling is done to catch HTTP errors, malformed JSON or move syntax and 
    the legality of the move to play.

    Args:
        board (chess.Board): Game instance used to verify legality of move
        url (str): URL of AI host to get move from
        retries (int, optional): How often to retry the request, if it failed. Defaults to 5.
        delay (int, optional): Delay between each retry in seconds. Defaults to 1.

    Returns:
        chess.Move: The move the AI wants to play. May be None but not illegal
    """
    # defined after REST API specifications
    payload = {
        "fen": board.fen(),
    }

    while retries > 0:
        retries -= 1

        # Catches connection issues on client and server
        try:
            response = requests.post(url, json=payload)
        except requests.ConnectionError as error:
            print(f"{error}! Retrying...")
            time.sleep(delay)
            continue

        try:
            # Fails, if 200 <= HTTP status code < 300 
            response.raise_for_status()
        except requests.HTTPError as error:
            print(f"{error}! Retrying...")
            time.sleep(delay)
            continue

        # Catch malformed JSON of incorrect server implementations
        try:
            move_san: str = response.json()["move"]
        except (requests.JSONDecodeError, KeyError):
            print("Couldn't decode JSON sent by server! Retrying...")
            time.sleep(delay)
            continue

        # Catch malformed Algebraic Notation
        try:
            move = board.parse_san(move_san)
        except ValueError:
            print("AI used invalid SAN notation for move!")
            time.sleep(delay)
            continue

        # Verifies legality of move
        if not board.is_legal(move):
            print("Move generated by AI is illegal!")
            time.sleep(delay)
            continue

        return move

    return None


def request_user_move(board: chess.Board):
    """
    Prompts the user to type his move in UCI.

    Args:
        board (chess.Board): Game instance to verify legality of move

    Returns:
        chess.Move: Move the player wants to play
    """
    while True:
        move = input("Insert your next move: ")
        # Catch malformed Universal Chess Interface notations
        try:
            move = board.parse_uci(move)
        except ValueError:
            print("Invalid UCI notation of move!")
            continue

        # Verifies legality of move
        if not board.is_legal(move):
            print("Illegal move!")
            continue

        return move
