#
# Player Extractor
# Get all players with a specific title on Chess.com and save them to a file.
# The code is single threaded because it must be; Chess.com doesn't allow simultanious requests but has no limit otherwise
#
# Chess.com documentation: https://www.chess.com/news/view/published-data-api#pubapi-endpoint-titled
#

import pathlib
import sys
import requests
import logging

# Chess.com Pub REST-API endpoint for filtering players by title
endpoint = "https://api.chess.com/pub/titled"

# Accepted titles by Chess.com; uncomment to select
titles = [
    "GM",
    #"WGM",
    #"IM",
    #"WIM",
    #"FM",
    #"WFM",
    #"NM",
    #"WNM",
    #"CM",
    #"WCM",
]

# Chess.com returns HTML instead of JSON if useragent isn't Postman
json_header = {
    'accept': 'application/json',
    'User-Agent': 'PostmanRuntime/10.21.0',
}

# Destination file for the player list
destination = "./rockyou.txt"
try:
    pathlib.Path(destination).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(destination).touch()
except OSError as error:
    logging.error(f"Couldn't create the destination file at: {destination}", exc_info=True)
    sys.exit()


def get_titled_players(title: str) -> list[str]:
    """Fetch the Chess.com names of players with a specific title

    Args:
        title (str): The title to filter for

    Returns:
        list[str]: The player names with the title
    """
    url = f"{endpoint}/{title}"
    
    try:
        request = requests.get(url=url, headers=json_header)
        request.raise_for_status()
        players: list[str] = request.json()["players"]
    except (requests.RequestException, KeyError) as error:
        logging.error(f"Failed to GET players with the {title} title!", exc_info=True)
        return []

    return players


def main():
    try:
        with open(destination, 'a') as file:
            for title in titles:
                for player in get_titled_players(title):
                    try:
                        file.write(player + "\n")
                    except OSError as exception:
                        logging.error(f"Couldn't write '{player}' to {destination}!", exc_info=True)
    except OSError as esception:
        logging.error(f"Couldn't open {destination}!", exc_info=True)


if __name__ == "__main__":
    print("Player Extractor:")
    print("Get all players with a specific title on Chess.com and save them to a file")
    main()
