import requests
import time
import os
import logging

save_dir = r"Games"
os.makedirs(save_dir, exist_ok=True)

# players to download games from
players = [
    "magnuscarlsen",
    "hikaru",
    "hansontwitch",
    "fabianocaruana",
    "chefshouse",
    "anishgiri",
    "lachesisq",
    "firouzja2003",
    "gmwso",
    "anishgiri",
    "sebastian",
    "sergeykarjakin",
    "anand",
    "tradjabov",
    "rpragchess",
    "viditchess",
    "chesswarrior7197",
    "lovevae",
    "ghandeevam2003",
    "vincentkeymer",
    "grischuk",
    "lyonbeast",
    "polish_fighter3000",
    "liemle",
    "gukeshdommaraju",
    "levonaronian",
]

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
    :param username: The user to download games from
    :return: A list of PGN strings
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
                if not game["rated"]:   # casual games
                    continue

                if game["time_class"] == "bullet":  # bullet = 1 minute games
                    continue

                if game["rules"] == "chess960": # ruleset that could mess with AI
                    continue

                filtered_games.append(game["pgn"])

            except KeyError as error:
                logging.error(f"Malformed game JSON in '{game}'", exc_info=True)
                continue

    return filtered_games


def save_game(user_name: str, unix_time: int, save_number: int, game_pgn: str) -> None:

    try:
        save_file = f"{unix_time}-{user_name}-{save_number}.pgn"
        file_path: str = os.path.join(save_dir, save_file)
        with open(file_path, "w") as file:
            file.write(game_pgn)

    except (IOError, OSError) as error:
        logging.error("An IO error occurred while saving a game!", exc_info=True)
        return


def process_games(player: str, start_time: int) -> int:
    print(f"Downloading {player}'s games...")
    games: list[str] = get_games_pgn(player)
    if not games:
        logging.warn(f"No games acquired for player {player}")
        return 1

    print(f"Saving {player}'s PGNs...")
    for i, game in enumerate(games):
        save_game(
            user_name=player,
            unix_time=start_time,
            save_number=i,
            game_pgn=game
        )

    return 0


def main(players: list[str]):
    current_time = int(time.time())
    for player in players:
        if process_games(player, start_time=current_time) != 0:
            print(f"An error occured while processing {player}'s games. Check the log!")


if __name__ == "__main__":
    main(players)
