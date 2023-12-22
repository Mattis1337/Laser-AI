import requests
import json

# Chess.com returns HTML instead of JSON if useragent isn't Postman
json_header = {
        'accept': 'application/json',
        'User-Agent': 'PostmanRuntime/10.16.0',
}


def get_archives(username):
    # creates the player's game archive URL
    url = f"https://api.chess.com/pub/player/{username}/games/archives"
    # downloads JSON as string
    archives = requests.get(url=url, headers=json_header).json()   # converts text to JSON
    return archives


def get_games(username):
    archives = get_archives(username=username)
    for archive in archives["archives"]:
        games = requests.get(url=archive, headers=json_header).json()


def save_game(game_pgn):
    print("TODO")
    # get player's archives
    # get pgn
    # save pgn


def save_games(username):
    get_games(get_archives(username))



get_games("Samuel")
# save_game("Samuel")
