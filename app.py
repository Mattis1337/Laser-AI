"""
The starting point of the CLI.
The parameters that are used to configure essential variables of the client are registered here.
After that some info is printed to STDOUT and lastly the game starts.
"""

# importing libraries
# import numpy as np
import argparse
import chess
import torch

# importing own files
import train_model
import cli


URL: str
FEN: str
COLOR: chess.Color
UNICODE: bool


def register_arguments():
    """
    Register arguments of script used to configure various mandatory options for the AI

    Returns:
        str: URL of the AI server
        str: Starting position of board
        chess.Color: Boolean representing the players color: white (True) or black (False)
        bool: Whether Unicode chars are enabled by user
    """
    # define arguments
    parser = argparse.ArgumentParser(
        description='Run the LaserAI client. ' +
                    'By default localhost and port 8000 are used for the server. ' +
                    'The usual chess starting position is used if no FEN code is provided explicitly.'
    )
    parser.add_argument('-i', '--ip', help='IP or domain of the server hosting LaserAI', default='127.0.0.1')
    parser.add_argument('-p', '--port', help='port the AI provider is listening to', default='8000')
    parser.add_argument('-f', '--fen', help='FEN code of the game to start against AI', default=chess.STARTING_FEN)
    parser.add_argument('-c', '--color', help='Whether the player should be white or black', default='white')
    parser.add_argument('-u', '--unicode', help='Enable unicode characters', action='store_true')

    # parse parameters
    args = parser.parse_args()
    url = f"http://{args.ip}:{args.port}/predict"
    fen = args.fen
    color: chess.Color
    match args.color.lower():
        case "white": color = chess.COLORS[0]
        case "black": color = chess.COLORS[1]
        case _:
            print("Color paramater may only be 'white' or 'black'!")
            exit(1)
    unicode=args.unicode

    return url, fen, color, unicode


def print_banner():
    """
    Print Laser-AI banner and important information about the clients configuration
    """
    print ( 
        r'  __          ___           _______. _______ .______               ___       __ ' + '\n' +
        r' |  |        /   \         /       ||   ____||   _  \             /   \     |  |' + '\n' +
        r' |  |       /  ^  \       |   (----`|  |__   |  |_)  |    ______ /  ^  \    |  |' + '\n' +
        r' |  |      /  /_\  \       \   \    |   __|  |      /    |______/  /_\  \   |  |' + '\n' +
        r' |  `----./  _____  \  .----)   |   |  |____ |  |\  \----.     /  _____  \  |  |' + '\n' +
        r' |_______/__/     \__\ |_______/    |_______|| _| `._____|    /__/     \__\ |__|' + '\n'
    )
    print(f"\n AI server is '{URL}'")
    print(f"\n Initial board state is '{FEN}'")


def main():
    """
    Command Line User Interface
    """

    while True:
        print('1) Start AI client' + '\n' +
              '2) Train AI locally' + '\n' +
              '3) Initialize new AI model')
        option = int(input('Pick an option 1-3: '))

        if option in range(1, 4):
            break
        print(f'Specified option {option} is invalid!')

    match option:
        case 1:
            # Starting the client
            cli.play_against_ai(FEN, UNICODE, ai_host=URL, ai_color=not COLOR)
        case 2:
            # Starting the training process
            train_model.train_chess_model()
        case 3:
            # Starting the initialization process
            train_model.init_new_model()


URL, FEN, COLOR, UNICODE = register_arguments()

if __name__ == "__main__":
    main()

print("Goodbye")
