import time
import requests

# use absolute paths! (or os.expand)
save_dir = r"/home/smuil/Downloads/Games"

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


def get_games_pgn(username):
    """
    Downloads all games in PGN format that a Chess.Com account has ever played in
    :param username: The user to download games from
    :return: A list of PGN strings
    """
    # builds the player's game archive URL which is an index of all games
    url = f"https://api.chess.com/pub/player/{username}/games/archives"
    archives = requests.get(url=url, headers=json_header).json()  # sends request and serializes the response string
    filtered_games = []
    # "archives" contains URLs of monthly game collections
    for archive in archives["archives"]:
        games = requests.get(url=archive, headers=json_header).json()["games"]  # 2d array of game representations
        for game in games:
            # apply game filters here
            if game["time_class"] == "bullet":
                continue
            filtered_games.append(game["pgn"])

    return filtered_games


def save_game(game_pgn):
    # /home/user/Downloads/Games/UnixTime.png
    file = open(f"{save_dir}/{time.time()}.pgn", "w")
    file.write(game_pgn)
    file.close()


def save_games(username):
    games = get_games_pgn(username)
    for game in games:
        save_game(game)


for player in players:
    save_games(player)
