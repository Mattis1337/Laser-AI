# importing libraries
# import numpy as np
import argparse
import chess

# importing own files
#import datasets
#import train_model
import cli

URL: str
FEN: str


def register_arguments():
    # define arguments
    parser = argparse.ArgumentParser(description='Run the LaserAI client')
    parser.add_argument('-i', '--ip', help='IP or domain of the server hosting LaserAI', default='127.0.0.1')
    parser.add_argument('-p', '--port', help='port the AI provider is listening to', default='8000')
    parser.add_argument('-f', '--fen', help='FEN code of the game to start against AI', default=chess.STARTING_FEN)

    # parse parameters
    args = parser.parse_args()
    url = f"http://{args.ip}:{args.port}/predict"
    fen = args.fen

    return url, fen


def print_banner():
    print ( 
        r' __          ___           _______. _______ .______               ___       __ ' + '\n' +
        r'|  |        /   \         /       ||   ____||   _  \             /   \     |  |' + '\n' +
        r'|  |       /  ^  \       |   (----`|  |__   |  |_)  |    ______ /  ^  \    |  |' + '\n' +
        r'|  |      /  /_\  \       \   \    |   __|  |      /    |______/  /_\  \   |  |' + '\n' +
        r'|  `----./  _____  \  .----)   |   |  |____ |  |\  \----.     /  _____  \  |  |' + '\n' +
        r'|_______/__/     \__\ |_______/    |_______|| _| `._____|    /__/     \__\ |__|' + '\n'
    )
    print(f"\n AI server is '{URL}'")
    print(f"\n Initial board state is '{FEN}'")


#dataset = datasets.init_chess_dataset(c.BLACK)
#train_model.train_chess_model(dataset, 20)

URL, FEN = register_arguments()
print_banner()
cli.play_against_ai(FEN, ai_host=URL, ai_color=chess.BLACK)

print("Goodbye")
