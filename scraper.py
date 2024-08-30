import requests
import time
import os

# use absolute paths! (or os.expand)
save_dir = r"Games"
os.makedirs(save_dir, exist_ok=True)

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
        print(f"Failed to GET archives of {username} with error: {error}")
        return filtered_games

    # "archives" contains URLs of monthly game collections
    for archive in archives:

        try:
            request = requests.get(url=archive, headers=json_header)
            request.raise_for_status()
            games: list[dict[str, str]] = request.json()["games"]

        except (requests.RequestException, KeyError) as error:
            print(f"Failed to GET games from '{archive}' with error: {error}")
            continue

        for game in games:

            try:
                # apply game filters here
                if game["time_class"] == "bullet":
                    continue
                filtered_games.append(game["pgn"])

            except KeyError as error:
                print(f"Malformed game JSON in '{game}'! {error}")
                continue

    return filtered_games


def save_game(game_pgn: str) -> None:

    try:
        timestamp = int(time.time())
        file_path: str = os.path.join(save_dir, f"{timestamp}.pgn")
        with open(file_path, "w") as file:
            file.write(game_pgn)    # /home/user/Downloads/Games/UnixTime.pgn

    except (IOError, OSError) as error:
        print("An IO error occurred while saving a game: " + error)


def process_games(player: str) -> None:
    games: list[str] = get_games_pgn(player)
    if not games or games.count() == 0:
        print(f"No games acquired for player {player}")
        return

    for game in games:
        save_game(game)


for player in players:
    process_games(player)
