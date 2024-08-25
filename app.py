# importing libraries
# import numpy as np
import argparse
import chess

# importing own files
#import datasets
#import train_model
import cli


def register_arguments():
    parser = argparse.ArgumentParser(description='Run the LaserAI client')
    parser.add_argument('-i', '--ip', help='IP or domain of the server hosting LaserAI', default='127.0.0.1')
    parser.add_argument('-p', '--port', help='port the AI provider is listening to', default='8000')
    args = parser.parse_args()
    return f"http://{args.ip}:{args.port}/predict" 


def print_banner():
    print ( 
        r' __          ___           _______. _______ .______               ___       __ ' + '\n' +
        r'|  |        /   \         /       ||   ____||   _  \             /   \     |  |' + '\n' +
        r'|  |       /  ^  \       |   (----`|  |__   |  |_)  |    ______ /  ^  \    |  |' + '\n' +
        r'|  |      /  /_\  \       \   \    |   __|  |      /    |______/  /_\  \   |  |' + '\n' +
        r'|  `----./  _____  \  .----)   |   |  |____ |  |\  \----.     /  _____  \  |  |' + '\n' +
        r'|_______/__/     \__\ |_______/    |_______|| _| `._____|    /__/     \__\ |__|' + '\n'
    )
    print("\nAI server is " + URL)


#dataset = datasets.init_chess_dataset(c.BLACK)
#train_model.train_chess_model(dataset, 20)

URL = register_arguments()
print_banner()
cli.play_against_ai(URL, ai_color=chess.BLACK)

print("Goodbye")
