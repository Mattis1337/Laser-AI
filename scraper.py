import requests
import time
import os
import logging

save_dir = r"Games"
os.makedirs(save_dir, exist_ok=True)

# Chess.com returns HTML instead of JSON if useragent isn't Postman
json_header = {
    'accept': 'application/json',
    'User-Agent': 'PostmanRuntime/10.21.0',
}

# define log file
log_dir = r"Logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "scraper.log")
if os.path.exists(log_file):
    os.remove(log_file)

# configure logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


def get_games_pgn(username: str) -> list[str]:
    """
    Downloads all public games that a Chess.com account has ever played as PGN

    Args:
        username (str): The user to download games from

    Returns:
        list[str]: A list of PGN strings
    """

    # builds the player's game archive URL which is an index of all games
    url = f"https://api.chess.com/pub/player/{username}/games/archives"
    filtered_games: list[str] = []
    
    try:
        request = requests.get(url=url, headers=json_header)
        request.raise_for_status()
        archives: list[str] = request.json()["archives"]

    except (requests.RequestException, KeyError) as error:
        logging.error(f"Failed to GET archives of {username}!", exc_info=True)
        return filtered_games

    # "archives" contains URLs of monthly game collections
    for archive in archives:

        try:
            request = requests.get(url=archive, headers=json_header)
            request.raise_for_status()
            games: list[dict[str, str]] = request.json()["games"]

        except (requests.RequestException, KeyError) as error:
            logging.error(f"Failed to GET games from '{archive}'!", exc_info=True)
            continue

        for game in games:

            try:
                # apply game filters here
                if game["rules"] != "chess":    # other rulesets could mess up AI
                    continue

                if not game["rated"]:   # casual games
                    continue

                if game["time_class"] == "bullet":  # bullet: < 3 minute games
                    continue

                filtered_games.append(game["pgn"])

            except KeyError as error:
                logging.error(f"Malformed game JSON in '{game}'", exc_info=True)
                continue

    return filtered_games


def save_game(unix_time: int, user_name: str, save_number: int, game_pgn: str) -> None:
    """
    Saves a string (PGN) to the Games directory in a file with the following format:
    "UNIXTIME-USERNAME-GAME#.pgn"
    The timestamp is useful when running the script multiple times
    to deleted old data which could lead to overfitting otherwise.
    The game number is just there to prevent overriding the PGN file. 

    Args:
        unix_time (int): timestamp used to distinguish between multiple runs
        user_name (str): the username the PGN belongs to
        save_number (int): save number preventing erasure of old PGNs
        game_pgn (str): The PGN data itself
    """

    try:
        save_file = f"{unix_time}-{user_name}-{save_number}.pgn"
        file_path: str = os.path.join(save_dir, save_file)
        with open(file_path, "w") as file:
            file.write(game_pgn)

    except OSError as error:
        logging.error("An IO error occurred while saving a game!", exc_info=True)
        return


def process_games(player: str, start_time: int) -> int:
    """
    Downloads public games of a Chess.com account
    and then saves them as individual PGN files

    Args:
        player (str): Chess.com username
        start_time (int): UNIX time to prepend to PGN names

    Returns:
        int: exit status
    """
    print(f"Downloading {player}'s games...")
    games: list[str] = get_games_pgn(player)
    if not games:   # make sure list is not None or empty
        logging.warning(f"No games acquired for player {player}")
        return 1

    print(f"Saving {player}'s PGNs...")
    for i, game in enumerate(games):
        save_game(
            unix_time=start_time,
            user_name=player,
            save_number=f"{i:05}",
            game_pgn=game
        )

    return 0


def get_players_from_file(file_path: str) -> list[str]:
    """
    Reads the Chess.com usernames from a file containing values separated by line breaks.

    Args:
        file_path (str): the file to parse

    Returns:
        list[str]: a list containing the file's values
    """
    players: list[str] = []
    try:
        with open(file_path, 'r') as file:
            # remove whitespaces and save strings between them into list
            players = [line.strip() for line in file]
    except OSError as error:
        logging.fatal(f"Couldn't read {file_path}!", exc_info=True)
        raise RuntimeError(f"Failed to read file: {file_path}") from error
    return players


def main(players: list[str]):
    current_time = int(time.time())
    for player in players:
        if process_games(player, start_time=current_time) != 0:
            print(f"An error occured while processing {player}'s games. Check the log!")


if __name__ == "__main__":
    # not handling RuntimeError is intentional
    # because the scraper can't work without a list; The Blacklist
    players = get_players_from_file("rockyou.txt")
    main(players)
